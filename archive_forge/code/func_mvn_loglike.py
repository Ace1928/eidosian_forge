import math
import numpy as np
from scipy import linalg, stats, special
from .linalg_decomp_1 import SvdArray
def mvn_loglike(x, sigma):
    """loglike multivariate normal

    assumes x is 1d, (nobs,) and sigma is 2d (nobs, nobs)

    brute force from formula
    no checking of correct inputs
    use of inv and log-det should be replace with something more efficient
    """
    sigmainv = linalg.inv(sigma)
    logdetsigma = np.log(np.linalg.det(sigma))
    nobs = len(x)
    llf = -np.dot(x, np.dot(sigmainv, x))
    llf -= nobs * np.log(2 * np.pi)
    llf -= logdetsigma
    llf *= 0.5
    return llf