# BBF202E Numerical Methods - Term Project - SVD Topic Modelling

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
# Standard stop words + bbc specific common terms (said, mr etc.)
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "is", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "as", "because", "until", "also",
    "said", "says", "say", "mr", "mrs", "ms", "would", "could", "one", "two",
    "year", "years", "told", "added", "made", "new", "people", "like", "back"
}
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z]+", " ", text)
    tokens = text.split()

    clean = []
    for t in tokens:
        if len(t) >= 3 and t not in STOP_WORDS:
            clean.append(t)
    return clean
def load_documents(data_path, max_docs):
    if not os.path.isdir(data_path):
        raise FileNotFoundError("Data path does not exist: " + data_path)
    documents = []
    labels = []
    categories = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    )
    if len(categories) == 0:
        raise FileNotFoundError("No category folders found inside " + data_path)
    for cat in categories:
        cat_dir = os.path.join(data_path, cat)
        files = sorted(f for f in os.listdir(cat_dir) if f.endswith(".txt"))
        files = files[:max_docs]
        for fname in files:
            fpath = os.path.join(cat_dir, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                text = fp.read()
            tokens = tokenize(text)
            if len(tokens) == 0:
                continue
            documents.append(tokens)
            labels.append(cat)
    if len(documents) == 0:
        raise ValueError("No non-empty documents were loaded.")
    return documents, labels
def build_tfidf_matrix(documents):
    vocab_set = set()
    for tokens in documents:
        vocab_set.update(tokens)
    vocab = sorted(vocab_set)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    m = len(vocab)
    n = len(documents)
    # Calculate raw counts
    counts = np.zeros((m, n), dtype=np.float64)
    for j, tokens in enumerate(documents):
        for t in tokens:
            counts[word_to_idx[t], j] += 1.0
    # TF computation
    doc_lengths = counts.sum(axis=0)
    doc_lengths[doc_lengths == 0] = 1.0
    tf = counts / doc_lengths
    # IDF computation
    df = (counts > 0).sum(axis=1)
    idf = np.log(n / np.maximum(df, 1))
    # Construct TF-IDF
    A = tf * idf[:, None]
    # Column-wise L2 normalization
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A = A / col_norms
    return A, vocab
def power_iteration(B, max_iter=2000, tol=1e-9, seed=None):
    rng = np.random.default_rng(seed)
    n = B.shape[0]
    v = rng.normal(size=n)
    v = v / np.linalg.norm(v)
    lam_old = 0.0
    lam = 0.0
    for i in range(max_iter):
        Bv = B @ v
        lam = float(v @ Bv) # Rayleigh quotient
        norm_Bv = np.linalg.norm(Bv)

        if norm_Bv < 1e-15:
            break

        v = Bv / norm_Bv
        if abs(lam - lam_old) < tol:
            # print(f"Converged at iteration {i}") # debug trace
            break
        lam_old = lam
    return lam, v
def eigen_decomposition_top_k(B, k):
    n = B.shape[0]
    eigenvalues = np.zeros(k)
    eigenvectors = np.zeros((n, k))
    B_def = B.astype(np.float64).copy()
    for i in range(k):
        # Dynamic seed definition to avoid orthogonality trap during deflation
        seed = i * 17 + 3
        lam, v = power_iteration(B_def, max_iter=2000, tol=1e-9, seed=seed)
        if lam < 0:
            lam = 0.0
        eigenvalues[i] = lam
        eigenvectors[:, i] = v
        # Wielandt Deflation step
        B_def = B_def - lam * np.outer(v, v)
    return eigenvalues, eigenvectors
def svd_from_scratch(A, k):
    B = A.T @ A
    eigenvalues, V = eigen_decomposition_top_k(B, k)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    sigma = np.sqrt(eigenvalues)
    # Reconstruct U matrix (handling tiny singular values)
    m = A.shape[0]
    U = np.zeros((m, k))
    AV = A @ V
    for i in range(k):
        if sigma[i] > 1e-12:
            U[:, i] = AV[:, i] / sigma[i]
        else:
            U[:, i] = 0.0
    VT = V.T
    return U, sigma, VT
def reconstruction_error(A, U, sigma, VT, k):
    Uk = U[:, :k]
    Sk = sigma[:k]
    VTk = VT[:k, :]

    A_k = (Uk * Sk) @ VTk
    diff = A - A_k
    return float(np.linalg.norm(diff, "fro"))
def plot_singular_values(sigma, output_path, total_energy=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(1, len(sigma) + 1), sigma, marker="o")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Singular value")
    axes[0].set_title("Singular value decay")
    axes[0].grid(True)
    energy = sigma ** 2
    if total_energy is None or total_energy <= 0:
        denom = energy.sum()
    else:
        denom = total_energy

    if denom == 0:
        cum = np.zeros_like(energy)
    else:
        cum = np.cumsum(energy) / denom
    axes[1].plot(range(1, len(sigma) + 1), cum, marker="o")
    axes[1].axhline(0.9, color="r", linestyle="--", label="90% energy")
    axes[1].set_xlabel("Rank k")
    axes[1].set_ylabel("Cumulative fraction of total energy")
    axes[1].set_title("Energy retention")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
def plot_reconstruction_errors(errors, output_path):
    ks = sorted(errors.keys())
    values = [errors[k] for k in ks]
    plt.figure(figsize=(7, 5))
    plt.plot(ks, values, marker="o")
    plt.xlabel("Rank k")
    plt.ylabel("||A - A_k||_F")
    plt.title("Reconstruction error vs rank")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
def save_topic_terms(U, vocab, output_path, num_topics=5, top_terms=10):
    os.makedirs(output_path, exist_ok=True)
    num_topics = min(num_topics, U.shape[1])

    for i in range(num_topics):
        weights = U[:, i]
        order = np.argsort(-np.abs(weights))[:top_terms] # sort by absolute weights
        fname = os.path.join(output_path, f"topic_{i + 1}_terms.txt")
        with open(fname, "w", encoding="utf-8") as fp:
            fp.write(f"Topic {i + 1} - top {top_terms} terms (sorted by |weight|)\n")
            fp.write("rank\tterm\tweight\n")
            for rank, idx in enumerate(order, start=1):
                fp.write(f"{rank}\t{vocab[idx]}\t{weights[idx]:.6f}\n")
def plot_document_embeddings(sigma, VT, labels, output_path):
    if VT.shape[0] < 2:
        raise ValueError("Need at least k=2 to draw a 2D embedding.")
    D = (sigma[:2, None] * VT[:2, :]).T
    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))
    labels_arr = np.array(labels)
    for color, lab in zip(colors, unique_labels):
        mask = labels_arr == lab
        plt.scatter(D[mask, 0], D[mask, 1],
                    color=color, label=lab, alpha=0.7, edgecolors="none")
    plt.xlabel("Concept 1")
    plt.ylabel("Concept 2")
    plt.title("2D document embedding (truncated SVD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()

# ----- Validation only (Task 3 Experiment in the project brief) ---------
# This function is NOT part of the SVD construction. It is only run when
# the --validate flag is given. The main pipeline (without the flag) never
# imports or calls numpy.linalg.svd.
def validation_against_numpy(A, sigma_ours):
    from numpy.linalg import svd as _np_svd   # local import, validation only
    _, s_ref, _ = _np_svd(A, full_matrices=False)
    k_eff = min(len(sigma_ours), len(s_ref))
    diff = np.abs(sigma_ours[:k_eff] - s_ref[:k_eff])
    max_diff = float(diff.max())
    print(f"  max |sigma_ours - sigma_numpy| over top-{k_eff} = {max_diff:.3e}")
    print(f"  mean abs diff                                  = {diff.mean():.3e}")
    return max_diff

def main():
    parser = argparse.ArgumentParser(
        description="SVD-based topic modeling on the BBC News dataset."
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the bbc/ folder containing category subfolders.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Folder where outputs will be written (created if missing).")
    parser.add_argument("--max_docs", type=int, default=100,
                        help="Maximum number of documents per category.")
    parser.add_argument("--k_svd", type=int, default=50,
                        help="Number of singular values/vectors to compute.")
    parser.add_argument("--validate", action="store_true",
                        help="Optional: compare sigma values against numpy.linalg.svd "
                             "(Task 3 Experiment in the project brief).")
    args = parser.parse_args()
    if args.max_docs < 1:
        raise ValueError("--max_docs must be at least 1.")
    if args.k_svd < 2:
        raise ValueError("--k_svd must be at least 2 (needed for the 2D plot).")
    os.makedirs(args.output_path, exist_ok=True)
    topic_dir = os.path.join(args.output_path, "topic_terms")
    os.makedirs(topic_dir, exist_ok=True)
    # -------- 1. Load and tokenize --------
    print("Loading documents...")
    documents, labels = load_documents(args.data_path, args.max_docs)
    print(f"  Loaded {len(documents)} documents from {len(set(labels))} categories.")
    # -------- 2. TF-IDF --------
    print("Building TF-IDF matrix...")
    A, vocab = build_tfidf_matrix(documents)
    print(f"  TF-IDF shape = {A.shape},  vocabulary size = {len(vocab)}")
    np.save(os.path.join(args.output_path, "tfidf_matrix.npy"), A)
    total_energy = float(np.linalg.norm(A, "fro") ** 2)
    k_max = min(A.shape) - 1
    k_svd = min(args.k_svd, k_max)
    if k_svd != args.k_svd:
        print(f"  Note: --k_svd reduced to {k_svd}.")
    # -------- 3. SVD from scratch --------
    print(f"Computing top-{k_svd} SVD from scratch (this may take a few minutes)...")
    U, sigma, VT = svd_from_scratch(A, k_svd)
    print(f"  Top-5 singular values: {np.round(sigma[:5], 4)}")
    plot_singular_values(
        sigma,
        os.path.join(args.output_path, "singular_value_decay.png"),
        total_energy=total_energy,
    )
    # -------- 4. Reconstruction errors --------
    print("Computing reconstruction errors...")
    target_ks = [2, 5, 10, 20, 50]
    errors = {}
    fractions = {}
    cum_sigma2 = np.cumsum(sigma ** 2)
    for kk in target_ks:
        if kk <= k_svd:
            err = reconstruction_error(A, U, sigma, VT, kk)
            errors[kk] = err
            fractions[kk] = float(cum_sigma2[kk - 1] / total_energy)
            print(f"  k={kk:>3d} : ||A - A_k||_F = {err:.6f}   "
                  f"({fractions[kk] * 100:.2f}% of total energy)")
    err_txt = os.path.join(args.output_path, "reconstruction_errors.txt")
    with open(err_txt, "w", encoding="utf-8") as fp:
        fp.write("rank_k\tfrobenius_error\tfraction_of_total_energy\n")
        for kk in sorted(errors.keys()):
            fp.write(f"{kk}\t{errors[kk]:.6f}\t{fractions[kk]:.6f}\n")
    plot_reconstruction_errors(
        errors, os.path.join(args.output_path, "reconstruction_errors.png")
    )
    # -------- Energy analysis --------
    fraction_top_k = cum_sigma2 / total_energy
    print(f"  ||A||_F^2 (total energy)            = {total_energy:.4f}")
    print(f"  Energy captured by top-{k_svd}          = {cum_sigma2[-1]:.4f}  "
          f"({fraction_top_k[-1] * 100:.2f}% of total)")
    if fraction_top_k[-1] >= 0.9:
        idx = int(np.searchsorted(fraction_top_k, 0.9))
        k90 = idx + 1
        print(f"  Minimum k that retains >= 90% of total energy: {k90}")
    else:
        print(f"  90% threshold NOT reached within k_svd={k_svd}.")
    # -------- 5. Topic terms and 2D embedding --------
    print("Saving top terms for the first 5 topics...")
    save_topic_terms(U, vocab, topic_dir, num_topics=5, top_terms=10)
    print("Plotting 2D document embedding...")
    plot_document_embeddings(
        sigma, VT, labels,
        os.path.join(args.output_path, "document_embeddings_2D.png")
    )
    # -------- Optional validation (Task 3 Experiment) --------
    if args.validate:
        print("Running validation against numpy.linalg.svd (Task 3 Experiment)...")
        validation_against_numpy(A, sigma)
    print("Done. All outputs written to:", args.output_path)
if __name__ == "__main__":
    main()
