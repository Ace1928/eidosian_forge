import numpy as np
from numba import cuda, complex64, int32, float64
from numba.cuda.testing import unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def cuconstRecAlign(A, B, C, D, E):
    Z = cuda.const.array_like(CONST_RECORD_ALIGN)
    i = cuda.grid(1)
    A[i] = Z[i]['a']
    B[i] = Z[i]['b']
    C[i] = Z[i]['x']
    D[i] = Z[i]['y']
    E[i] = Z[i]['z']