import numpy as np
from collections import namedtuple
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
def test_namedunituple(self):

    @cuda.jit
    def f(r, x):
        r[0] = x.x
        r[1] = x.y
    Point = namedtuple('Point', ('x', 'y'))
    x = Point(1, 2)
    r = np.zeros(len(x), dtype=np.int64)
    f[1, 1](r, x)
    self.assertEqual(r[0], x.x)
    self.assertEqual(r[1], x.y)