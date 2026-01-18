import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
@guvectorize('()->()', identity=1)
def double(x, out):
    out[:] = x * 2