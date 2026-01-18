import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def mquantiles_cimj(data, prob=[0.25, 0.5, 0.75], alpha=0.05, axis=None):
    """
    Computes the alpha confidence interval for the selected quantiles of the
    data, with Maritz-Jarrett estimators.

    Parameters
    ----------
    data : ndarray
        Data array.
    prob : sequence, optional
        Sequence of quantiles to compute.
    alpha : float, optional
        Confidence level of the intervals.
    axis : int or None, optional
        Axis along which to compute the quantiles.
        If None, use a flattened array.

    Returns
    -------
    ci_lower : ndarray
        The lower boundaries of the confidence interval.  Of the same length as
        `prob`.
    ci_upper : ndarray
        The upper boundaries of the confidence interval.  Of the same length as
        `prob`.

    """
    alpha = min(alpha, 1 - alpha)
    z = norm.ppf(1 - alpha / 2.0)
    xq = mstats.mquantiles(data, prob, alphap=0, betap=0, axis=axis)
    smj = mjci(data, prob, axis=axis)
    return (xq - z * smj, xq + z * smj)