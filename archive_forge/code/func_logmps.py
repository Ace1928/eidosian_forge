import numpy as np
from scipy import stats, optimize, special
def logmps(params, xsorted, dist):
    """calculate negative log of Product-of-Spacings

    Parameters
    ----------
    params : array_like, tuple ?
        parameters of the distribution funciton
    xsorted : array_like
        data that is already sorted
    dist : instance of a distribution class
        only cdf method is used

    Returns
    -------
    mps : float
        negative log of Product-of-Spacings


    Notes
    -----
    MPS definiton from JKB page 233
    """
    xcdf = np.r_[0.0, dist.cdf(xsorted, *params), 1.0]
    D = np.diff(xcdf)
    return -np.log(D).mean()