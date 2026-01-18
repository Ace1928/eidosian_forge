import numpy as np
from scipy import stats, optimize, special
def fit_mps(dist, data, x0=None):
    """Estimate distribution parameters with Maximum Product-of-Spacings

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
    x : ndarray
        estimates for the parameters of the distribution given the data,
        including loc and scale


    """
    xsorted = np.sort(data)
    if x0 is None:
        x0 = getstartparams(dist, xsorted)
    args = (xsorted, dist)
    print(x0)
    return optimize.fmin(logmps, x0, args=args)