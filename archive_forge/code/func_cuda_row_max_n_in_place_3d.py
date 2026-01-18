from __future__ import annotations
from math import ceil, isnan
from packaging.version import Version
import numba
import numpy as np
from numba import cuda
@cuda.jit
def cuda_row_max_n_in_place_3d(ret, other):
    """CUDA equivalent of row_max_n_in_place_3d.
    """
    ny, nx, _n = ret.shape
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        _cuda_row_max_n_impl(ret[y, x], other[y, x])