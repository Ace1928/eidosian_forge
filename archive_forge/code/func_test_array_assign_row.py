from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_array_assign_row(self):
    arr = np.arange(5 * 7).reshape(5, 7)
    darr = cuda.to_device(arr)
    _400 = np.full(shape=5, fill_value=400)
    arr[:, 2] = _400
    darr[:, 2] = _400
    np.testing.assert_array_equal(darr.copy_to_host(), arr)