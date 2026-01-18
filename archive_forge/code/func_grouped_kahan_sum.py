from __future__ import annotations
from typing import (
import numba
from numba.extending import register_jitable
import numpy as np
from pandas.core._numba.kernels.shared import is_monotonic_increasing
@register_jitable
def grouped_kahan_sum(values: np.ndarray, result_dtype: np.dtype, labels: npt.NDArray[np.intp], ngroups: int) -> tuple[np.ndarray, npt.NDArray[np.int64], np.ndarray, npt.NDArray[np.int64], np.ndarray]:
    N = len(labels)
    nobs_arr = np.zeros(ngroups, dtype=np.int64)
    comp_arr = np.zeros(ngroups, dtype=values.dtype)
    consecutive_counts = np.zeros(ngroups, dtype=np.int64)
    prev_vals = np.zeros(ngroups, dtype=values.dtype)
    output = np.zeros(ngroups, dtype=result_dtype)
    for i in range(N):
        lab = labels[i]
        val = values[i]
        if lab < 0:
            continue
        sum_x = output[lab]
        nobs = nobs_arr[lab]
        compensation_add = comp_arr[lab]
        num_consecutive_same_value = consecutive_counts[lab]
        prev_value = prev_vals[lab]
        nobs, sum_x, compensation_add, num_consecutive_same_value, prev_value = add_sum(val, nobs, sum_x, compensation_add, num_consecutive_same_value, prev_value)
        output[lab] = sum_x
        consecutive_counts[lab] = num_consecutive_same_value
        prev_vals[lab] = prev_value
        comp_arr[lab] = compensation_add
        nobs_arr[lab] = nobs
    return (output, nobs_arr, comp_arr, consecutive_counts, prev_vals)