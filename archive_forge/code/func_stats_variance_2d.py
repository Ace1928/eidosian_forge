import warnings
from collections.abc import Sequence
from copy import copy as _copy
from copy import deepcopy as _deepcopy
import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.stats.mstats import mquantiles
from xarray import apply_ufunc
from .. import _log
from ..utils import conditional_jit, conditional_vect, conditional_dask
from .density_utils import histogram as _histogram
def stats_variance_2d(data, ddof=0, axis=1):
    if data.ndim == 1:
        return stats_variance_1d(data, ddof=ddof)
    a_a, b_b = data.shape
    if axis == 1:
        var = np.zeros(a_a)
        for i in range(a_a):
            var[i] = stats_variance_1d(data[i], ddof=ddof)
    else:
        var = np.zeros(b_b)
        for i in range(b_b):
            var[i] = stats_variance_1d(data[:, i], ddof=ddof)
    return var