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
def find_repeats(arr):
    """Find repeats in arr and return a tuple (repeats, repeat_count).

    The input is cast to float64. Masked values are discarded.

    Parameters
    ----------
    arr : sequence
        Input array. The array is flattened if it is not 1D.

    Returns
    -------
    repeats : ndarray
        Array of repeated values.
    counts : ndarray
        Array of counts.

    Examples
    --------
    >>> from scipy.stats import mstats
    >>> mstats.find_repeats([2, 1, 2, 3, 2, 2, 5])
    (array([2.]), array([4]))

    In the above example, 2 repeats 4 times.

    >>> mstats.find_repeats([[10, 20, 1, 2], [5, 5, 4, 4]])
    (array([4., 5.]), array([2, 2]))

    In the above example, both 4 and 5 repeat 2 times.

    """
    compr = np.asarray(ma.compressed(arr), dtype=np.float64)
    try:
        need_copy = np.may_share_memory(compr, arr)
    except AttributeError:
        need_copy = False
    if need_copy:
        compr = compr.copy()
    return _find_repeats(compr)