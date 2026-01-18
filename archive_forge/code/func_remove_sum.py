from __future__ import annotations
from typing import (
import numba
from numba.extending import register_jitable
import numpy as np
from pandas.core._numba.kernels.shared import is_monotonic_increasing
@numba.jit(nopython=True, nogil=True, parallel=False)
def remove_sum(val: Any, nobs: int, sum_x: Any, compensation: Any) -> tuple[int, Any, Any]:
    if not np.isnan(val):
        nobs -= 1
        y = -val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t
    return (nobs, sum_x, compensation)