import os
from scipy.special._precompute.utils import lagrange_inversion
def compute_g(n):
    """g_k from DLMF 5.11.3/5.11.5"""
    a = compute_a(2 * n)
    g = [mp.sqrt(2) * mp.rf(0.5, k) * a[2 * k] for k in range(n)]
    return g