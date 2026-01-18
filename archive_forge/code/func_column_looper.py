from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
@numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
def column_looper(values: np.ndarray, start: np.ndarray, end: np.ndarray, min_periods: int, *args):
    result = np.empty((values.shape[0], len(start)), dtype=result_dtype)
    na_positions = {}
    for i in numba.prange(values.shape[0]):
        output, na_pos = func(values[i], result_dtype, start, end, min_periods, *args)
        result[i] = output
        if len(na_pos) > 0:
            na_positions[i] = np.array(na_pos)
    return (result, na_positions)