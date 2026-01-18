import numpy as np
from scipy.special import comb
def se_cov(cov):
    """
    get standard deviation from covariance matrix

    just a shorthand function np.sqrt(np.diag(cov))

    Parameters
    ----------
    cov : array_like, square
        covariance matrix

    Returns
    -------
    std : ndarray
        standard deviation from diagonal of cov
    """
    return np.sqrt(np.diag(cov))