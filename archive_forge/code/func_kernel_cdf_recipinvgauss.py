import numpy as np
from scipy import special, stats
def kernel_cdf_recipinvgauss(x, sample, bw):
    m = 1 / (x - bw)
    lam = 1 / bw
    return stats.recipinvgauss.sf(sample, m / lam, scale=1 / lam)