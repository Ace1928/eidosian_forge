import numpy as np
from scipy import special
from scipy.special import gammaln
def llnormscale(scale, y, loc):
    return np.log(stats.norm.pdf(y, loc=loc, scale=scale))