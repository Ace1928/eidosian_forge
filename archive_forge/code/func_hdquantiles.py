import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def hdquantiles(data, prob=list([0.25, 0.5, 0.75]), axis=None, var=False):
    """
    Computes quantile estimates with the Harrell-Davis method.

    The quantile estimates are calculated as a weighted linear combination
    of order statistics.

    Parameters
    ----------
    data : array_like
        Data array.
    prob : sequence, optional
        Sequence of probabilities at which to compute the quantiles.
    axis : int or None, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.
    var : bool, optional
        Whether to return the variance of the estimate.

    Returns
    -------
    hdquantiles : MaskedArray
        A (p,) array of quantiles (if `var` is False), or a (2,p) array of
        quantiles and variances (if `var` is True), where ``p`` is the
        number of quantiles.

    See Also
    --------
    hdquantiles_sd

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.mstats import hdquantiles
    >>>
    >>> # Sample data
    >>> data = np.array([1.2, 2.5, 3.7, 4.0, 5.1, 6.3, 7.0, 8.2, 9.4])
    >>>
    >>> # Probabilities at which to compute quantiles
    >>> probabilities = [0.25, 0.5, 0.75]
    >>>
    >>> # Compute Harrell-Davis quantile estimates
    >>> quantile_estimates = hdquantiles(data, prob=probabilities)
    >>>
    >>> # Display the quantile estimates
    >>> for i, quantile in enumerate(probabilities):
    ...     print(f"{int(quantile * 100)}th percentile: {quantile_estimates[i]}")
    25th percentile: 3.1505820231763066 # may vary
    50th percentile: 5.194344084883956
    75th percentile: 7.430626414674935

    """

    def _hd_1D(data, prob, var):
        """Computes the HD quantiles for a 1D array. Returns nan for invalid data."""
        xsorted = np.squeeze(np.sort(data.compressed().view(ndarray)))
        n = xsorted.size
        hd = np.empty((2, len(prob)), float64)
        if n < 2:
            hd.flat = np.nan
            if var:
                return hd
            return hd[0]
        v = np.arange(n + 1) / float(n)
        betacdf = beta.cdf
        for i, p in enumerate(prob):
            _w = betacdf(v, (n + 1) * p, (n + 1) * (1 - p))
            w = _w[1:] - _w[:-1]
            hd_mean = np.dot(w, xsorted)
            hd[0, i] = hd_mean
            hd[1, i] = np.dot(w, (xsorted - hd_mean) ** 2)
        hd[0, prob == 0] = xsorted[0]
        hd[0, prob == 1] = xsorted[-1]
        if var:
            hd[1, prob == 0] = hd[1, prob == 1] = np.nan
            return hd
        return hd[0]
    data = ma.array(data, copy=False, dtype=float64)
    p = np.array(prob, copy=False, ndmin=1)
    if axis is None or data.ndim == 1:
        result = _hd_1D(data, p, var)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_hd_1D, axis, data, p, var)
    return ma.fix_invalid(result, copy=False)