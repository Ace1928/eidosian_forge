import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import (
def skewtest(a, axis=0, alternative='two-sided'):
    """
    Tests whether the skew is different from the normal distribution.

    Parameters
    ----------
    a : array_like
        The data to be tested
    axis : int or None, optional
       Axis along which statistics are calculated. Default is 0.
       If None, compute over the whole array `a`.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the skewness of the distribution underlying the sample
          is different from that of the normal distribution (i.e. 0)
        * 'less': the skewness of the distribution underlying the sample
          is less than that of the normal distribution
        * 'greater': the skewness of the distribution underlying the sample
          is greater than that of the normal distribution

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : array_like
        The computed z-score for this test.
    pvalue : array_like
        A p-value for the hypothesis test

    Notes
    -----
    For more details about `skewtest`, see `scipy.stats.skewtest`.

    """
    a, axis = _chk_asarray(a, axis)
    if axis is None:
        a = a.ravel()
        axis = 0
    b2 = skew(a, axis)
    n = a.count(axis)
    if np.min(n) < 8:
        raise ValueError('skewtest is not valid with less than 8 samples; %i samples were given.' % np.min(n))
    y = b2 * ma.sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)))
    beta2 = 3.0 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9))
    W2 = -1 + ma.sqrt(2 * (beta2 - 1))
    delta = 1 / ma.sqrt(0.5 * ma.log(W2))
    alpha = ma.sqrt(2.0 / (W2 - 1))
    y = ma.where(y == 0, 1, y)
    Z = delta * ma.log(y / alpha + ma.sqrt((y / alpha) ** 2 + 1))
    return SkewtestResult(*scipy.stats._stats_py._normtest_finish(Z, alternative))