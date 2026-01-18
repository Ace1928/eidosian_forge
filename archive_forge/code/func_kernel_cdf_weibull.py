import numpy as np
from scipy import special, stats
def kernel_cdf_weibull(x, sample, bw):
    return stats.weibull_min.sf(sample, 1 / bw, scale=x / special.gamma(1 + bw))