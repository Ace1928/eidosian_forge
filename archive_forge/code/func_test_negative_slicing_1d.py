from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_negative_slicing_1d(self):
    arr = np.arange(10)
    darr = cuda.to_device(arr)
    for i, j in product(range(-10, 10), repeat=2):
        np.testing.assert_array_equal(arr[i:j], darr[i:j].copy_to_host())