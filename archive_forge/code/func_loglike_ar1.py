import math
import numpy as np
from scipy import linalg, stats, special
from .linalg_decomp_1 import SvdArray
def loglike_ar1(x, rho):
    """loglikelihood of AR(1) process, as a test case

    sigma_u partially hard coded

    Greene chapter 12 eq. (12-31)
    """
    x = np.asarray(x)
    u = np.r_[x[0], x[1:] - rho * x[:-1]]
    sigma_u2 = 2 * (1 - rho ** 2)
    loglik = 0.5 * (-(u ** 2).sum(0) / sigma_u2 + np.log(1 - rho ** 2) - x.shape[0] * (np.log(2 * np.pi) + np.log(sigma_u2)))
    return loglik