from __future__ import annotations
from typing import TYPE_CHECKING
import numba
import numpy as np
from pandas.core._numba.kernels.shared import is_monotonic_increasing
from pandas.core._numba.kernels.sum_ import grouped_kahan_sum
@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_mean(values: np.ndarray, result_dtype: np.dtype, labels: npt.NDArray[np.intp], ngroups: int, min_periods: int) -> tuple[np.ndarray, list[int]]:
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
            result = np.nan
        result /= nobs
        output[lab] = result
    na_pos = [0 for i in range(0)]
    return (output, na_pos)