from __future__ import annotations
from math import ceil, isnan
from packaging.version import Version
import numba
import numpy as np
from numba import cuda
@cuda.jit(device=True)
def cuda_mutex_lock(mutex, index):
    while cuda.atomic.compare_and_swap(mutex, 0, 1) != 0:
        pass
    cuda.threadfence()