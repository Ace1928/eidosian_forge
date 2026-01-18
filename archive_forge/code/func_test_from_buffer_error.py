import array
import numpy as np
from numba import jit, njit
import numba.core.typing.cffi_utils as cffi_support
from numba.core import types, errors
from numba.tests.support import TestCase, skip_unless_cffi
import numba.tests.cffi_usecases as mod
import unittest
def test_from_buffer_error(self):
    pyfunc = mod.vector_sin_float32
    cfunc = jit(nopython=True)(pyfunc)
    x = np.arange(10).astype(np.float32)[::2]
    y = np.zeros_like(x)
    with self.assertRaises(errors.TypingError) as raises:
        cfunc(x, y)
    self.assertIn('from_buffer() unsupported on non-contiguous buffers', str(raises.exception))