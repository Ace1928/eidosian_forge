import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_assign_array_to_nested_2d(self):
    src = (np.arange(6) + 1).astype(np.int16).reshape((3, 2))
    got = np.zeros(2, dtype=nested_array2_dtype)
    expected = np.zeros(2, dtype=nested_array2_dtype)
    pyfunc = assign_array_to_nested_2d
    kernel = cuda.jit(pyfunc)
    kernel[1, 1](got[0], src)
    pyfunc(expected[0], src)
    np.testing.assert_array_equal(expected, got)