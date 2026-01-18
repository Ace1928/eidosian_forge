import array
import numpy as np
from numba import jit, njit
import numba.core.typing.cffi_utils as cffi_support
from numba.core import types, errors
from numba.tests.support import TestCase, skip_unless_cffi
import numba.tests.cffi_usecases as mod
import unittest
def test_from_buffer_struct(self):
    n = 10
    x = np.arange(n) + np.arange(n * 2, n * 3) * 1j
    y = np.zeros(n)
    real_cfunc = jit(nopython=True)(mod.vector_extract_real)
    real_cfunc(x, y)
    np.testing.assert_equal(x.real, y)
    imag_cfunc = jit(nopython=True)(mod.vector_extract_imag)
    imag_cfunc(x, y)
    np.testing.assert_equal(x.imag, y)