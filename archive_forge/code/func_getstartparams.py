import numpy as np
from scipy import stats, optimize, special
def getstartparams(dist, data):
    """get starting values for estimation of distribution parameters

    Parameters
    ----------
    dist : distribution instance
        the distribution instance needs to have either a method fitstart
        or an attribute numargs
    data : ndarray
        data for which preliminary estimator or starting value for
        parameter estimation is desired

    Returns
    -------
    x0 : ndarray
        preliminary estimate or starting value for the parameters of
        the distribution given the data, including loc and scale

    """
    if hasattr(dist, 'fitstart'):
        x0 = dist.fitstart(data)
    elif np.isfinite(dist.a):
        x0 = np.r_[[1.0] * dist.numargs, data.min() - 1, 1.0]
    else:
        x0 = np.r_[[1.0] * dist.numargs, data.mean() - 1, 1.0]
    return x0