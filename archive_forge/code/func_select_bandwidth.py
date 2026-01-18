import numpy as np
from scipy.stats import scoreatpercentile
from statsmodels.compat.pandas import Substitution
from statsmodels.sandbox.nonparametric import kernels
@Substitution(', '.join(sorted(bandwidth_funcs.keys())))
def select_bandwidth(x, bw, kernel):
    """
    Selects bandwidth for a selection rule bw

    this is a wrapper around existing bandwidth selection rules

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    bw : str
        name of bandwidth selection rule, currently supported are:
        %s
    kernel : not used yet

    Returns
    -------
    bw : float
        The estimate of the bandwidth
    """
    bw = bw.lower()
    if bw not in bandwidth_funcs:
        raise ValueError('Bandwidth %s not understood' % bw)
    bandwidth = bandwidth_funcs[bw](x, kernel)
    if np.any(bandwidth == 0):
        err = 'Selected KDE bandwidth is 0. Cannot estimate density. Either provide the bandwidth during initialization or use an alternative method.'
        raise RuntimeError(err)
    else:
        return bandwidth