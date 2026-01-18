import array
import numpy as np
from numba import jit, njit
import numba.core.typing.cffi_utils as cffi_support
from numba.core import types, errors
from numba.tests.support import TestCase, skip_unless_cffi
import numba.tests.cffi_usecases as mod
import unittest
def test_from_buffer_pyarray(self):
    pyfunc = mod.vector_sin_float32
    cfunc = jit(nopython=True)(pyfunc)
    x = array.array('f', range(10))
    y = array.array('f', [0] * len(x))
    self.check_vector_sin(cfunc, x, y)