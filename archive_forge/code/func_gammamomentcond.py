import numpy as np
from scipy import stats, optimize, special
def gammamomentcond(distfn, params, mom2, quantile=None):
    """estimate distribution parameters based method of moments (mean,
    variance) for distributions with 1 shape parameter and fixed loc=0.

    Returns
    -------
    cond : function

    Notes
    -----
    first test version, quantile argument not used

    """

    def cond(params):
        alpha, scale = params
        mom2s = distfn.stats(alpha, 0.0, scale)
        return np.array(mom2) - mom2s
    return cond