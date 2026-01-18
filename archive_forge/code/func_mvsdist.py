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
def mvsdist(data):
    """
    'Frozen' distributions for mean, variance, and standard deviation of data.

    Parameters
    ----------
    data : array_like
        Input array. Converted to 1-D using ravel.
        Requires 2 or more data-points.

    Returns
    -------
    mdist : "frozen" distribution object
        Distribution object representing the mean of the data.
    vdist : "frozen" distribution object
        Distribution object representing the variance of the data.
    sdist : "frozen" distribution object
        Distribution object representing the standard deviation of the data.

    See Also
    --------
    bayes_mvs

    Notes
    -----
    The return values from ``bayes_mvs(data)`` is equivalent to
    ``tuple((x.mean(), x.interval(0.90)) for x in mvsdist(data))``.

    In other words, calling ``<dist>.mean()`` and ``<dist>.interval(0.90)``
    on the three distribution objects returned from this function will give
    the same results that are returned from `bayes_mvs`.

    References
    ----------
    T.E. Oliphant, "A Bayesian perspective on estimating mean, variance, and
    standard-deviation from data", https://scholarsarchive.byu.edu/facpub/278,
    2006.

    Examples
    --------
    >>> from scipy import stats
    >>> data = [6, 9, 12, 7, 8, 8, 13]
    >>> mean, var, std = stats.mvsdist(data)

    We now have frozen distribution objects "mean", "var" and "std" that we can
    examine:

    >>> mean.mean()
    9.0
    >>> mean.interval(0.95)
    (6.6120585482655692, 11.387941451734431)
    >>> mean.std()
    1.1952286093343936

    """
    x = ravel(data)
    n = len(x)
    if n < 2:
        raise ValueError('Need at least 2 data-points.')
    xbar = x.mean()
    C = x.var()
    if n > 1000:
        mdist = distributions.norm(loc=xbar, scale=math.sqrt(C / n))
        sdist = distributions.norm(loc=math.sqrt(C), scale=math.sqrt(C / (2.0 * n)))
        vdist = distributions.norm(loc=C, scale=math.sqrt(2.0 / n) * C)
    else:
        nm1 = n - 1
        fac = n * C / 2.0
        val = nm1 / 2.0
        mdist = distributions.t(nm1, loc=xbar, scale=math.sqrt(C / nm1))
        sdist = distributions.gengamma(val, -2, scale=math.sqrt(fac))
        vdist = distributions.invgamma(val, scale=fac)
    return (mdist, vdist, sdist)