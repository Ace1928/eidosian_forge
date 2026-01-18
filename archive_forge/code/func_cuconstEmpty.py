import numpy as np
from numba import cuda, complex64, int32, float64
from numba.cuda.testing import unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def cuconstEmpty(A):
    C = cuda.const.array_like(CONST_EMPTY)
    i = cuda.grid(1)
    A[i] = len(C)