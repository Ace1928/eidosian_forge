import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def rsh(data, points=None):
    """
    Evaluates Rosenblatt's shifted histogram estimators for each data point.

    Rosenblatt's estimator is a centered finite-difference approximation to the
    derivative of the empirical cumulative distribution function.

    Parameters
    ----------
    data : sequence
        Input data, should be 1-D. Masked values are ignored.
    points : sequence or None, optional
        Sequence of points where to evaluate Rosenblatt shifted histogram.
        If None, use the data.

    """
    data = ma.array(data, copy=False)
    if points is None:
        points = data
    else:
        points = np.array(points, copy=False, ndmin=1)
    if data.ndim != 1:
        raise AttributeError('The input array should be 1D only !')
    n = data.count()
    r = idealfourths(data, axis=None)
    h = 1.2 * (r[-1] - r[0]) / n ** (1.0 / 5)
    nhi = (data[:, None] <= points[None, :] + h).sum(0)
    nlo = (data[:, None] < points[None, :] - h).sum(0)
    return (nhi - nlo) / (2.0 * n * h)