import numpy as np
from scipy import special, stats
def kernel_cdf_invgauss(x, sample, bw):
    m = x
    lam = 1 / bw
    return stats.invgauss.sf(sample, m / lam, scale=lam)