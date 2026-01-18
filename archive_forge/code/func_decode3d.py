import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
def decode3d(encoded, size_max=1024):
    coord_dtype = cupy.int32 if size_max < 2 ** 31 else cupy.int64
    x = cupy.empty_like(encoded, dtype=coord_dtype)
    y = cupy.empty_like(x)
    z = cupy.empty_like(x)
    kern = _get_decode3d_kernel(size_max)
    kern(encoded, x, y, z)
    return (x, y, z)