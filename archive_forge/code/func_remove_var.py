from __future__ import annotations
from typing import TYPE_CHECKING
import numba
import numpy as np
from pandas.core._numba.kernels.shared import is_monotonic_increasing
@numba.jit(nopython=True, nogil=True, parallel=False)
def remove_var(val: float, nobs: int, mean_x: float, ssqdm_x: float, compensation: float) -> tuple[int, float, float, float]:
    if not np.isnan(val):
        nobs -= 1
        if nobs:
            prev_mean = mean_x - compensation
            y = val - compensation
            t = y - mean_x
            compensation = t + mean_x - y
            delta = t
            mean_x -= delta / nobs
            ssqdm_x -= (val - prev_mean) * (val - mean_x)
        else:
            mean_x = 0
            ssqdm_x = 0
    return (nobs, mean_x, ssqdm_x, compensation)