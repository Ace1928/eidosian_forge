import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def hdquantiles_sd(data, prob=list([0.25, 0.5, 0.75]), axis=None):
    """
    The standard error of the Harrell-Davis quantile estimates by jackknife.

    Parameters
    ----------
    data : array_like
        Data array.
    prob : sequence, optional
        Sequence of quantiles to compute.
    axis : int, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.

    Returns
    -------
    hdquantiles_sd : MaskedArray
        Standard error of the Harrell-Davis quantile estimates.

    See Also
    --------
    hdquantiles

    """

    def _hdsd_1D(data, prob):
        """Computes the std error for 1D arrays."""
        xsorted = np.sort(data.compressed())
        n = len(xsorted)
        hdsd = np.empty(len(prob), float64)
        if n < 2:
            hdsd.flat = np.nan
        vv = np.arange(n) / float(n - 1)
        betacdf = beta.cdf
        for i, p in enumerate(prob):
            _w = betacdf(vv, n * p, n * (1 - p))
            w = _w[1:] - _w[:-1]
            mx_ = np.zeros_like(xsorted)
            mx_[1:] = np.cumsum(w * xsorted[:-1])
            mx_[:-1] += np.cumsum(w[::-1] * xsorted[:0:-1])[::-1]
            hdsd[i] = np.sqrt(mx_.var() * (n - 1))
        return hdsd
    data = ma.array(data, copy=False, dtype=float64)
    p = np.array(prob, copy=False, ndmin=1)
    if axis is None:
        result = _hdsd_1D(data, p)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_hdsd_1D, axis, data, p)
    return ma.fix_invalid(result, copy=False).ravel()