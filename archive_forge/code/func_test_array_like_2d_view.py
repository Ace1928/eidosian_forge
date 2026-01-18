import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
from numba import config, cuda
def test_array_like_2d_view(self):
    shape = (10, 12)
    view = np.zeros(shape)[::2, ::2]
    d_view = cuda.device_array(shape)[::2, ::2]
    for like_func in ARRAY_LIKE_FUNCTIONS:
        with self.subTest(like_func=like_func):
            self._test_array_like_view(like_func, view, d_view)