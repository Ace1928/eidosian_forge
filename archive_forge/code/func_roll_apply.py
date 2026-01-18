from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.core.util.numba_ import jit_user_function
@numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
def roll_apply(values: np.ndarray, begin: np.ndarray, end: np.ndarray, minimum_periods: int, *args: Any) -> np.ndarray:
    result = np.empty(len(begin))
    for i in numba.prange(len(result)):
        start = begin[i]
        stop = end[i]
        window = values[start:stop]
        count_nan = np.sum(np.isnan(window))
        if len(window) - count_nan >= minimum_periods:
            result[i] = numba_func(window, *args)
        else:
            result[i] = np.nan
    return result