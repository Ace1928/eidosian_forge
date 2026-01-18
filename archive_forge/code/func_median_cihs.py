import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def median_cihs(data, alpha=0.05, axis=None):
    """
    Computes the alpha-level confidence interval for the median of the data.

    Uses the Hettmasperger-Sheather method.

    Parameters
    ----------
    data : array_like
        Input data. Masked values are discarded. The input should be 1D only,
        or `axis` should be set to None.
    alpha : float, optional
        Confidence level of the intervals.
    axis : int or None, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.

    Returns
    -------
    median_cihs
        Alpha level confidence interval.

    """

    def _cihs_1D(data, alpha):
        data = np.sort(data.compressed())
        n = len(data)
        alpha = min(alpha, 1 - alpha)
        k = int(binom._ppf(alpha / 2.0, n, 0.5))
        gk = binom.cdf(n - k, n, 0.5) - binom.cdf(k - 1, n, 0.5)
        if gk < 1 - alpha:
            k -= 1
            gk = binom.cdf(n - k, n, 0.5) - binom.cdf(k - 1, n, 0.5)
        gkk = binom.cdf(n - k - 1, n, 0.5) - binom.cdf(k, n, 0.5)
        I = (gk - 1 + alpha) / (gk - gkk)
        lambd = (n - k) * I / float(k + (n - 2 * k) * I)
        lims = (lambd * data[k] + (1 - lambd) * data[k - 1], lambd * data[n - k - 1] + (1 - lambd) * data[n - k])
        return lims
    data = ma.array(data, copy=False)
    if axis is None:
        result = _cihs_1D(data, alpha)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_cihs_1D, axis, data, alpha)
    return result