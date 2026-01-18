import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
def test_guvectorize_decor(self):
    gufunc = guvectorize([void(float32[:, :], float32[:, :], float32[:, :])], '(m,n),(n,p)->(m,p)', target=self.target)(matmulcore)
    self.check_matmul_gufunc(gufunc)