import numpy as np
from scipy import special
from scipy.special import gammaln
def llnorm(y, loc, scale):
    return np.log(stats.norm.pdf(y, loc=loc, scale=scale))