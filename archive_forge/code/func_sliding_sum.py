from __future__ import annotations
from typing import (
import numba
from numba.extending import register_jitable
import numpy as np
from pandas.core._numba.kernels.shared import is_monotonic_increasing
@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_sum(values: np.ndarray, result_dtype: np.dtype, start: np.ndarray, end: np.ndarray, min_periods: int) -> tuple[np.ndarray, list[int]]:
    dtype = values.dtype
    na_val: object = np.nan
    if dtype.kind == 'i':
        na_val = 0
    N = len(start)
    nobs = 0
    sum_x = 0
    compensation_add = 0
    compensation_remove = 0
    na_pos = []
    is_monotonic_increasing_bounds = is_monotonic_increasing(start) and is_monotonic_increasing(end)
    output = np.empty(N, dtype=result_dtype)
    for i in range(N):
        s = start[i]
        e = end[i]
        if i == 0 or not is_monotonic_increasing_bounds:
            prev_value = values[s]
            num_consecutive_same_value = 0
            for j in range(s, e):
                val = values[j]
                nobs, sum_x, compensation_add, num_consecutive_same_value, prev_value = add_sum(val, nobs, sum_x, compensation_add, num_consecutive_same_value, prev_value)
        else:
            for j in range(start[i - 1], s):
                val = values[j]
                nobs, sum_x, compensation_remove = remove_sum(val, nobs, sum_x, compensation_remove)
            for j in range(end[i - 1], e):
                val = values[j]
                nobs, sum_x, compensation_add, num_consecutive_same_value, prev_value = add_sum(val, nobs, sum_x, compensation_add, num_consecutive_same_value, prev_value)
        if nobs == 0 == min_periods:
            result: object = 0
        elif nobs >= min_periods:
            if num_consecutive_same_value >= nobs:
                result = prev_value * nobs
            else:
                result = sum_x
        else:
            result = na_val
            if dtype.kind == 'i':
                na_pos.append(i)
        output[i] = result
        if not is_monotonic_increasing_bounds:
            nobs = 0
            sum_x = 0
            compensation_remove = 0
    return (output, na_pos)