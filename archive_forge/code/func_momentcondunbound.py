import numpy as np
from scipy import stats, optimize, special
def momentcondunbound(distfn, params, mom2, quantile=None):
    """moment conditions for estimating distribution parameters using method
    of moments, uses mean, variance and one quantile for distributions
    with 1 shape parameter.

    Returns
    -------
    difference : ndarray
        difference between theoretical and empirical moments and quantiles

    """
    shape, loc, scale = params
    mom2diff = np.array(distfn.stats(shape, loc, scale)) - mom2
    if quantile is not None:
        pq, xq = quantile
        cdfdiff = distfn.cdf(xq, shape, loc, scale) - pq
        return np.concatenate([mom2diff, cdfdiff[:1]])
    return mom2diff