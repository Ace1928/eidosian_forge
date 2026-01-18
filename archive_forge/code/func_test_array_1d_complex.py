import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def test_array_1d_complex(self, flags=force_pyobj_flags):
    self.run_array_1d(types.complex128, np.arange(5.0) * 1j, flags)