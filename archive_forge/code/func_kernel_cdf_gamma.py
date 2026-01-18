import numpy as np
from scipy import special, stats
def kernel_cdf_gamma(x, sample, bw):
    cdfi = stats.gamma.sf(sample, x / bw + 1, scale=bw)
    return cdfi