import os
from scipy.special._precompute.utils import lagrange_inversion
def compute_d(K, N):
    """d_{k, n} from DLMF 8.12.12"""
    M = N + 2 * K
    d0 = [-mp.mpf(1) / 3]
    alpha = compute_alpha(M + 2)
    for n in range(1, M):
        d0.append((n + 2) * alpha[n + 2])
    d = [d0]
    g = compute_g(K)
    for k in range(1, K):
        dk = []
        for n in range(M - 2 * k):
            dk.append((-1) ** k * g[k] * d[0][n] + (n + 2) * d[k - 1][n + 2])
        d.append(dk)
    for k in range(K):
        d[k] = d[k][:N]
    return d