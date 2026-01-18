from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.core.util.numba_ import jit_user_function
@numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
def roll_table(values: np.ndarray, begin: np.ndarray, end: np.ndarray, minimum_periods: int, *args: Any):
    result = np.empty((len(begin), values.shape[1]))
    min_periods_mask = np.empty(result.shape)
    for i in numba.prange(len(result)):
        start = begin[i]
        stop = end[i]
        window = values[start:stop]
        count_nan = np.sum(np.isnan(window), axis=0)
        sub_result = numba_func(window, *args)
        nan_mask = len(window) - count_nan >= minimum_periods
        min_periods_mask[i, :] = nan_mask
        result[i, :] = sub_result
    result = np.where(min_periods_mask, result, np.nan)
    return result