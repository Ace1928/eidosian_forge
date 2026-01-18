import math
import itertools
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import types
from numba import cuda
from numba.tests.complex_usecases import (real_usecase, imag_usecase,
from numba.np import numpy_support
def test_atomic_on_imag(self):

    @cuda.jit
    def atomic_add_one_j(values):
        i = cuda.grid(1)
        cuda.atomic.add(values.imag, i, 1)
    N = 32
    arr1 = np.arange(N) + np.arange(N) * 1j
    arr2 = arr1.copy()
    atomic_add_one_j[1, N](arr2)
    np.testing.assert_equal(arr1 + 1j, arr2)