import numpy as np
from scipy import special, stats
def kernel_cdf_lognorm(x, sample, bw):
    bw_ = np.sqrt(4 * np.log(1 + bw))
    return stats.lognorm.sf(sample, bw_, scale=x)