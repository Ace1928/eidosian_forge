import array
import numpy as np
from numba import jit, njit
import numba.core.typing.cffi_utils as cffi_support
from numba.core import types, errors
from numba.tests.support import TestCase, skip_unless_cffi
import numba.tests.cffi_usecases as mod
import unittest
def test_from_buffer_float64(self):
    self._test_from_buffer_numpy_array(mod.vector_sin_float64, np.float64)