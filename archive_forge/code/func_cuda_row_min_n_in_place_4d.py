from __future__ import annotations
from math import ceil, isnan
from packaging.version import Version
import numba
import numpy as np
from numba import cuda
@cuda.jit
def cuda_row_min_n_in_place_4d(ret, other):
    """CUDA equivalent of row_min_n_in_place_4d.
    """
    ny, nx, ncat, _n = ret.shape
    x, y, cat = cuda.grid(3)
    if x < nx and y < ny and (cat < ncat):
        _cuda_row_min_n_impl(ret[y, x, cat], other[y, x, cat])