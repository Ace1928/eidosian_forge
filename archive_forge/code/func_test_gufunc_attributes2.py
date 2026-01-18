import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
def test_gufunc_attributes2(self):

    @guvectorize('(),()->()')
    def add(x, y, res):
        res[0] = x + y
    self.assertIsNone(add.signature)
    a = np.array([1, 2, 3, 4])
    b = np.array([4, 3, 2, 1])
    res = np.array([0, 0, 0, 0])
    add(a, b, res)
    self.assertPreciseEqual(res, np.array([5, 5, 5, 5]))
    self.assertIsNone(add.signature)
    self.assertEqual(add.reduce(a), 10)
    self.assertPreciseEqual(add.accumulate(a), np.array([1, 3, 6, 10]))
    self.assertPreciseEqual(add.outer([0, 1], [1, 2]), np.array([[1, 2], [2, 3]]))
    self.assertPreciseEqual(add.reduceat(a, [0, 2]), np.array([3, 7]))
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2])
    add.at(x, [0, 1], y)
    self.assertPreciseEqual(x, np.array([2, 4, 3, 4]))