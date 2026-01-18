import numpy as np
from collections import namedtuple
from itertools import product
from numba import vectorize
from numba import cuda, int32, float32, float64
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
import unittest
def test_manual_transfer(self):

    @vectorize(signatures, target='cuda')
    def vector_add(a, b):
        return a + b
    n = 10
    x = np.arange(n, dtype=np.int32)
    dx = cuda.to_device(x)
    expected = x + x
    actual = vector_add(x, dx).copy_to_host()
    np.testing.assert_equal(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)