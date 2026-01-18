import array
import numpy as np
from numba import jit, njit
import numba.core.typing.cffi_utils as cffi_support
from numba.core import types, errors
from numba.tests.support import TestCase, skip_unless_cffi
import numba.tests.cffi_usecases as mod
import unittest
def test_from_buffer_numpy_multi_array(self):
    c1 = np.array([1, 2], order='C', dtype=np.float32)
    c1_zeros = np.zeros_like(c1)
    c2 = np.array([[1, 2], [3, 4]], order='C', dtype=np.float32)
    c2_zeros = np.zeros_like(c2)
    f1 = np.array([1, 2], order='F', dtype=np.float32)
    f1_zeros = np.zeros_like(f1)
    f2 = np.array([[1, 2], [3, 4]], order='F', dtype=np.float32)
    f2_zeros = np.zeros_like(f2)
    f2_copy = f2.copy('K')
    pyfunc = mod.vector_sin_float32
    cfunc = jit(nopython=True)(pyfunc)
    self.check_vector_sin(cfunc, c1, c1_zeros)
    cfunc(c2, c2_zeros)
    sin_c2 = np.sin(c2)
    sin_c2[1] = [0, 0]
    np.testing.assert_allclose(c2_zeros, sin_c2)
    self.check_vector_sin(cfunc, f1, f1_zeros)
    with self.assertRaises(errors.TypingError) as raises:
        cfunc(f2, f2_zeros)
    np.testing.assert_allclose(f2, f2_copy)
    self.assertIn('from_buffer() only supports multidimensional arrays with C layout', str(raises.exception))