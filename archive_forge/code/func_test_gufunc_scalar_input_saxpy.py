import numpy as np
from numba import guvectorize, cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
def test_gufunc_scalar_input_saxpy(self):

    @guvectorize(['void(float32, float32[:], float32[:], float32[:])'], '(),(t),(t)->(t)', target='cuda')
    def saxpy(a, x, y, out):
        for i in range(out.shape[0]):
            out[i] = a * x[i] + y[i]
    A = np.float32(2)
    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    Y = np.arange(10, dtype=np.float32).reshape(5, 2)
    out = saxpy(A, X, Y)
    for j in range(5):
        for i in range(2):
            exp = A * X[j, i] + Y[j, i]
            self.assertTrue(exp == out[j, i])
    X = np.arange(10, dtype=np.float32)
    Y = np.arange(10, dtype=np.float32)
    out = saxpy(A, X, Y)
    for j in range(10):
        exp = A * X[j] + Y[j]
        self.assertTrue(exp == out[j], (exp, out[j]))
    A = np.arange(5, dtype=np.float32)
    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    Y = np.arange(10, dtype=np.float32).reshape(5, 2)
    out = saxpy(A, X, Y)
    for j in range(5):
        for i in range(2):
            exp = A[j] * X[j, i] + Y[j, i]
            self.assertTrue(exp == out[j, i], (exp, out[j, i]))