from __future__ import annotations
import math
import warnings
from collections import namedtuple
import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
from ._ansari_swilk_statistics import gscale, swilk
from . import _stats_py
from ._fit import FitResult
from ._stats_py import find_repeats, _normtest_finish, SignificanceResult
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._hypotests import _get_wilcoxon_distr
from ._axis_nan_policy import _axis_nan_policy_factory
@_axis_nan_policy_factory(lambda x: x, result_to_tuple=lambda x: (x,), n_outputs=1, default_axis=None)
def kstat(data, n=2):
    """
    Return the nth k-statistic (1<=n<=4 so far).

    The nth k-statistic k_n is the unique symmetric unbiased estimator of the
    nth cumulant kappa_n.

    Parameters
    ----------
    data : array_like
        Input array. Note that n-D input gets flattened.
    n : int, {1, 2, 3, 4}, optional
        Default is equal to 2.

    Returns
    -------
    kstat : float
        The nth k-statistic.

    See Also
    --------
    kstatvar : Returns an unbiased estimator of the variance of the k-statistic
    moment : Returns the n-th central moment about the mean for a sample.

    Notes
    -----
    For a sample size n, the first few k-statistics are given by:

    .. math::

        k_{1} = \\mu
        k_{2} = \\frac{n}{n-1} m_{2}
        k_{3} = \\frac{ n^{2} } {(n-1) (n-2)} m_{3}
        k_{4} = \\frac{ n^{2} [(n + 1)m_{4} - 3(n - 1) m^2_{2}]} {(n-1) (n-2) (n-3)}

    where :math:`\\mu` is the sample mean, :math:`m_2` is the sample
    variance, and :math:`m_i` is the i-th sample central moment.

    References
    ----------
    http://mathworld.wolfram.com/k-Statistic.html

    http://mathworld.wolfram.com/Cumulant.html

    Examples
    --------
    >>> from scipy import stats
    >>> from numpy.random import default_rng
    >>> rng = default_rng()

    As sample size increases, n-th moment and n-th k-statistic converge to the
    same number (although they aren't identical). In the case of the normal
    distribution, they converge to zero.

    >>> for n in [2, 3, 4, 5, 6, 7]:
    ...     x = rng.normal(size=10**n)
    ...     m, k = stats.moment(x, 3), stats.kstat(x, 3)
    ...     print("%.3g %.3g %.3g" % (m, k, m-k))
    -0.631 -0.651 0.0194  # random
    0.0282 0.0283 -8.49e-05
    -0.0454 -0.0454 1.36e-05
    7.53e-05 7.53e-05 -2.26e-09
    0.00166 0.00166 -4.99e-09
    -2.88e-06 -2.88e-06 8.63e-13
    """
    if n > 4 or n < 1:
        raise ValueError('k-statistics only supported for 1<=n<=4')
    n = int(n)
    S = np.zeros(n + 1, np.float64)
    data = ravel(data)
    N = data.size
    if N == 0:
        raise ValueError('Data input must not be empty')
    if np.isnan(np.sum(data)):
        return np.nan
    for k in range(1, n + 1):
        S[k] = np.sum(data ** k, axis=0)
    if n == 1:
        return S[1] * 1.0 / N
    elif n == 2:
        return (N * S[2] - S[1] ** 2.0) / (N * (N - 1.0))
    elif n == 3:
        return (2 * S[1] ** 3 - 3 * N * S[1] * S[2] + N * N * S[3]) / (N * (N - 1.0) * (N - 2.0))
    elif n == 4:
        return (-6 * S[1] ** 4 + 12 * N * S[1] ** 2 * S[2] - 3 * N * (N - 1.0) * S[2] ** 2 - 4 * N * (N + 1) * S[1] * S[3] + N * N * (N + 1) * S[4]) / (N * (N - 1.0) * (N - 2.0) * (N - 3.0))
    else:
        raise ValueError('Should not be here.')