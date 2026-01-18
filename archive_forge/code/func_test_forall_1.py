import numpy as np
from numba import cuda
import unittest
from numba.cuda.testing import CUDATestCase
def test_forall_1(self):
    arr = np.arange(11)
    orig = arr.copy()
    foo.forall(arr.size)(arr)
    np.testing.assert_array_almost_equal(arr, orig + 1)