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
def kstatvar(data, n=2):
    """Return an unbiased estimator of the variance of the k-statistic.

    See `kstat` for more details of the k-statistic.

    Parameters
    ----------
    data : array_like
        Input array. Note that n-D input gets flattened.
    n : int, {1, 2}, optional
        Default is equal to 2.

    Returns
    -------
    kstatvar : float
        The nth k-statistic variance.

    See Also
    --------
    kstat : Returns the n-th k-statistic.
    moment : Returns the n-th central moment about the mean for a sample.

    Notes
    -----
    The variances of the first few k-statistics are given by:

    .. math::

        var(k_{1}) = \\frac{\\kappa^2}{n}
        var(k_{2}) = \\frac{\\kappa^4}{n} + \\frac{2\\kappa^2_{2}}{n - 1}
        var(k_{3}) = \\frac{\\kappa^6}{n} + \\frac{9 \\kappa_2 \\kappa_4}{n - 1} +
                     \\frac{9 \\kappa^2_{3}}{n - 1} +
                     \\frac{6 n \\kappa^3_{2}}{(n-1) (n-2)}
        var(k_{4}) = \\frac{\\kappa^8}{n} + \\frac{16 \\kappa_2 \\kappa_6}{n - 1} +
                     \\frac{48 \\kappa_{3} \\kappa_5}{n - 1} +
                     \\frac{34 \\kappa^2_{4}}{n-1} +
                     \\frac{72 n \\kappa^2_{2} \\kappa_4}{(n - 1) (n - 2)} +
                     \\frac{144 n \\kappa_{2} \\kappa^2_{3}}{(n - 1) (n - 2)} +
                     \\frac{24 (n + 1) n \\kappa^4_{2}}{(n - 1) (n - 2) (n - 3)}
    """
    data = ravel(data)
    N = len(data)
    if n == 1:
        return kstat(data, n=2) * 1.0 / N
    elif n == 2:
        k2 = kstat(data, n=2)
        k4 = kstat(data, n=4)
        return (2 * N * k2 ** 2 + (N - 1) * k4) / (N * (N + 1))
    else:
        raise ValueError('Only n=1 or n=2 supported.')