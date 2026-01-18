import numpy as np
from scipy import stats, optimize, special
def momentcondunboundls(distfn, params, mom2, quantile=None, shape=None):
    """moment conditions for estimating loc and scale of a distribution
    with method of moments using either 2 quantiles or 2 moments (not both).

    Returns
    -------
    difference : ndarray
        difference between theoretical and empirical moments or quantiles

    """
    loc, scale = params
    mom2diff = np.array(distfn.stats(shape, loc, scale)) - mom2
    if quantile is not None:
        pq, xq = quantile
        cdfdiff = distfn.cdf(xq, shape, loc, scale) - pq
        return cdfdiff
    return mom2diff