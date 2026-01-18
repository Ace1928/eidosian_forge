from numba import cuda
import numpy as np
from numba.cuda.testing import CUDATestCase
from numba.tests.support import override_config
import unittest
def test_device_jit(self):

    @cuda.jit(device=True)
    def mapper(args):
        a, b, c = args
        return a + b + c

    @cuda.jit(device=True)
    def reducer(a, b):
        return a + b

    @cuda.jit
    def driver(A, B):
        i = cuda.grid(1)
        if i < B.size:
            args = (A[i], A[i] + B[i], B[i])
            B[i] = reducer(mapper(args), 1)
    A = np.arange(100, dtype=np.float32)
    B = np.arange(100, dtype=np.float32)
    Acopy = A.copy()
    Bcopy = B.copy()
    driver[1, 100](A, B)
    np.testing.assert_allclose(Acopy + Acopy + Bcopy + Bcopy + 1, B)