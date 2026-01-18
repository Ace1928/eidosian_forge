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
@_rename_parameter('mode', 'method')
def ks_1samp(x, cdf, args=(), alternative='two-sided', method='auto'):
    """
    Computes the Kolmogorov-Smirnov test on one sample of masked values.

    Missing values in `x` are discarded.

    Parameters
    ----------
    x : array_like
        a 1-D array of observations of random variables.
    cdf : str or callable
        If a string, it should be the name of a distribution in `scipy.stats`.
        If a callable, that callable is used to calculate the cdf.
    args : tuple, sequence, optional
        Distribution parameters, used if `cdf` is a string.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Indicates the alternative hypothesis.  Default is 'two-sided'.
    method : {'auto', 'exact', 'asymp'}, optional
        Defines the method used for calculating the p-value.
        The following options are available (default is 'auto'):

          * 'auto' : use 'exact' for small size arrays, 'asymp' for large
          * 'exact' : use approximation to exact distribution of test statistic
          * 'asymp' : use asymptotic distribution of test statistic

    Returns
    -------
    d : float
        Value of the Kolmogorov Smirnov test
    p : float
        Corresponding p-value.

    """
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(alternative.lower()[0], alternative)
    return scipy.stats._stats_py.ks_1samp(x, cdf, args=args, alternative=alternative, method=method)