from __future__ import annotations
from typing import (
import numba
from numba.extending import register_jitable
import numpy as np
from pandas.core._numba.kernels.shared import is_monotonic_increasing
@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_sum(values: np.ndarray, result_dtype: np.dtype, labels: npt.NDArray[np.intp], ngroups: int, min_periods: int) -> tuple[np.ndarray, list[int]]:
    na_pos = []
    output, nobs_arr, comp_arr, consecutive_counts, prev_vals = grouped_kahan_sum(values, result_dtype, labels, ngroups)
    for lab in range(ngroups):
        nobs = nobs_arr[lab]
        num_consecutive_same_value = consecutive_counts[lab]
        prev_value = prev_vals[lab]
        sum_x = output[lab]
        if nobs >= min_periods:
            if num_consecutive_same_value >= nobs:
                result = prev_value * nobs
            else:
                result = sum_x
        else:
            result = sum_x
            na_pos.append(lab)
        output[lab] = result
    return (output, na_pos)