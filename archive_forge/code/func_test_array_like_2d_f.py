import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
from numba import config, cuda
def test_array_like_2d_f(self):
    d_a = cuda.device_array((10, 12), order='F')
    for like_func in ARRAY_LIKE_FUNCTIONS:
        with self.subTest(like_func=like_func):
            self._test_array_like_same(like_func, d_a)