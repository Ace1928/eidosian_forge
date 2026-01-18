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
def kurtosistest(a, axis=0, alternative='two-sided'):
    """
    Tests whether a dataset has normal kurtosis

    Parameters
    ----------
    a : array_like
        array of the sample data
    axis : int or None, optional
       Axis along which to compute test. Default is 0. If None,
       compute over the whole array `a`.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the kurtosis of the distribution underlying the sample
          is different from that of the normal distribution
        * 'less': the kurtosis of the distribution underlying the sample
          is less than that of the normal distribution
        * 'greater': the kurtosis of the distribution underlying the sample
          is greater than that of the normal distribution

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : array_like
        The computed z-score for this test.
    pvalue : array_like
        The p-value for the hypothesis test

    Notes
    -----
    For more details about `kurtosistest`, see `scipy.stats.kurtosistest`.

    """
    a, axis = _chk_asarray(a, axis)
    n = a.count(axis=axis)
    if np.min(n) < 5:
        raise ValueError('kurtosistest requires at least 5 observations; %i observations were given.' % np.min(n))
    if np.min(n) < 20:
        warnings.warn('kurtosistest only valid for n>=20 ... continuing anyway, n=%i' % np.min(n), stacklevel=2)
    b2 = kurtosis(a, axis, fisher=False)
    E = 3.0 * (n - 1) / (n + 1)
    varb2 = 24.0 * n * (n - 2.0) * (n - 3) / ((n + 1) * (n + 1.0) * (n + 3) * (n + 5))
    x = (b2 - E) / ma.sqrt(varb2)
    sqrtbeta1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * np.sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
    A = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + np.sqrt(1 + 4.0 / sqrtbeta1 ** 2))
    term1 = 1 - 2.0 / (9.0 * A)
    denom = 1 + x * ma.sqrt(2 / (A - 4.0))
    if np.ma.isMaskedArray(denom):
        denom[denom == 0.0] = masked
    elif denom == 0.0:
        denom = masked
    term2 = np.ma.where(denom > 0, ma.power((1 - 2.0 / A) / denom, 1 / 3.0), -ma.power(-(1 - 2.0 / A) / denom, 1 / 3.0))
    Z = (term1 - term2) / np.sqrt(2 / (9.0 * A))
    return KurtosistestResult(*scipy.stats._stats_py._normtest_finish(Z, alternative))