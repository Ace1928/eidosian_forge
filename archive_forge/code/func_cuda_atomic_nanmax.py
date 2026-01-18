from __future__ import annotations
from math import ceil, isnan
from packaging.version import Version
import numba
import numpy as np
from numba import cuda
@cuda.jit(device=True)
def cuda_atomic_nanmax(ary, idx, val):
    return cuda.atomic.max(ary, idx, val)