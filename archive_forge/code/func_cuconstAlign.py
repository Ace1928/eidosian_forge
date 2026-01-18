import numpy as np
from numba import cuda, complex64, int32, float64
from numba.cuda.testing import unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def cuconstAlign(z):
    a = cuda.const.array_like(CONST3BYTES)
    b = cuda.const.array_like(CONST1D)
    i = cuda.grid(1)
    z[i] = a[i] + b[i]