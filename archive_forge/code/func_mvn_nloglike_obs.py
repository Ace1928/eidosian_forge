import math
import numpy as np
from scipy import linalg, stats, special
from .linalg_decomp_1 import SvdArray
def mvn_nloglike_obs(x, sigma):
    """loglike multivariate normal

    assumes x is 1d, (nobs,) and sigma is 2d (nobs, nobs)

    brute force from formula
    no checking of correct inputs
    use of inv and log-det should be replace with something more efficient
    """
    sigmainv = linalg.inv(sigma)
    cholsigmainv = linalg.cholesky(sigmainv)
    x_whitened = np.dot(cholsigmainv, x)
    logdetsigma = np.log(np.linalg.det(sigma))
    sigma2 = 1.0
    llike = 0.5 * (np.log(sigma2) - 2.0 * np.log(np.diagonal(cholsigmainv)) + x_whitened ** 2 / sigma2 + np.log(2 * np.pi))
    return (llike, x_whitened ** 2)