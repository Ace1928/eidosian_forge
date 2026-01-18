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
def kstest(data1, data2, args=(), alternative='two-sided', method='auto'):
    """

    Parameters
    ----------
    data1 : array_like
    data2 : str, callable or array_like
    args : tuple, sequence, optional
        Distribution parameters, used if `data1` or `data2` are strings.
    alternative : str, as documented in stats.kstest
    method : str, as documented in stats.kstest

    Returns
    -------
    tuple of (K-S statistic, probability)

    """
    return scipy.stats._stats_py.kstest(data1, data2, args, alternative=alternative, method=method)