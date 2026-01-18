import numpy as np
from scipy import special, stats
def kernel_cdf_invgamma(x, sample, bw):
    return stats.invgamma.sf(sample, 1 / bw + 1, scale=x / bw)