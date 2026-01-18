import numpy as np
from numpy import ma
from scipy import stats
def plotting_positions_w1d(data, weights=None, alpha=0.4, beta=0.4, method='notnormed'):
    """Weighted plotting positions (or empirical percentile points) for the data.

    observations are weighted and the plotting positions are defined as
    (ws-alpha)/(n-alpha-beta), where:
        - ws is the weighted rank order statistics or cumulative weighted sum,
          normalized to n if method is "normed"
        - n is the number of values along the given axis if method is "normed"
          and total weight otherwise
        - alpha and beta are two parameters.

    wtd.quantile in R package Hmisc seems to use the "notnormed" version.
    notnormed coincides with unweighted segment in example, drop "normed" version ?


    See Also
    --------
    plotting_positions : unweighted version that works also with more than one
        dimension and has other options
    """
    x = np.atleast_1d(data)
    if x.ndim > 1:
        raise ValueError('currently implemented only for 1d')
    if weights is None:
        weights = np.ones(x.shape)
    else:
        weights = np.array(weights, float, copy=False, ndmin=1)
        if weights.shape != x.shape:
            raise ValueError('if weights is given, it needs to be the sameshape as data')
    n = len(x)
    xargsort = x.argsort()
    ws = weights[xargsort].cumsum()
    res = np.empty(x.shape)
    if method == 'normed':
        res[xargsort] = (1.0 * ws / ws[-1] * n - alpha) / (n + 1.0 - alpha - beta)
    else:
        res[xargsort] = (1.0 * ws - alpha) / (ws[-1] + 1.0 - alpha - beta)
    return res