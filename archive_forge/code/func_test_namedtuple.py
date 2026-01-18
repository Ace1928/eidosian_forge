import numpy as np
from collections import namedtuple
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
def test_namedtuple(self):

    @cuda.jit
    def f(r1, r2, x):
        r1[0] = x.x
        r1[1] = x.y
        r2[0] = x.r
    Point = namedtuple('Point', ('x', 'y', 'r'))
    x = Point(1, 2, 2.236)
    r1 = np.zeros(2, dtype=np.int64)
    r2 = np.zeros(1, dtype=np.float64)
    f[1, 1](r1, r2, x)
    self.assertEqual(r1[0], x.x)
    self.assertEqual(r1[1], x.y)
    self.assertEqual(r2[0], x.r)