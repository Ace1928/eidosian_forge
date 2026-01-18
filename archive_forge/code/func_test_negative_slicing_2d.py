from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_negative_slicing_2d(self):
    arr = np.arange(12).reshape(3, 4)
    darr = cuda.to_device(arr)
    for x, y, w, s in product(range(-4, 4), repeat=4):
        np.testing.assert_array_equal(arr[x:y, w:s], darr[x:y, w:s].copy_to_host())