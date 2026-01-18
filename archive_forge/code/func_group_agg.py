from __future__ import annotations
import functools
import inspect
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.core.util.numba_ import (
@numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
def group_agg(values: np.ndarray, index: np.ndarray, begin: np.ndarray, end: np.ndarray, num_columns: int, *args: Any) -> np.ndarray:
    assert len(begin) == len(end)
    num_groups = len(begin)
    result = np.empty((num_groups, num_columns))
    for i in numba.prange(num_groups):
        group_index = index[begin[i]:end[i]]
        for j in numba.prange(num_columns):
            group = values[begin[i]:end[i], j]
            result[i, j] = numba_func(group, group_index, *args)
    return result