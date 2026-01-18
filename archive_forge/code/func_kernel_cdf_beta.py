import numpy as np
from scipy import special, stats
def kernel_cdf_beta(x, sample, bw):
    return stats.beta.sf(sample, x / bw + 1, (1 - x) / bw + 1)