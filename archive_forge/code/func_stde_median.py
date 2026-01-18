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
def stde_median(data, axis=None):
    """Returns the McKean-Schrader estimate of the standard error of the sample
    median along the given axis. masked values are discarded.

    Parameters
    ----------
    data : ndarray
        Data to trim.
    axis : {None,int}, optional
        Axis along which to perform the trimming.
        If None, the input array is first flattened.

    """

    def _stdemed_1D(data):
        data = np.sort(data.compressed())
        n = len(data)
        z = 2.5758293035489004
        k = int(np.round((n + 1) / 2.0 - z * np.sqrt(n / 4.0), 0))
        return (data[n - k] - data[k - 1]) / (2.0 * z)
    data = ma.array(data, copy=False, subok=True)
    if axis is None:
        return _stdemed_1D(data)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, but got data.ndim = %d" % data.ndim)
        return ma.apply_along_axis(_stdemed_1D, axis, data)