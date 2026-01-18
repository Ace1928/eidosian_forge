import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_literal_variable(self):
    arr = np.array([1, 2], dtype=recordtype2)
    pyfunc = set_field1
    jitfunc = njit(pyfunc)
    self.assertEqual(pyfunc(arr[0].copy()), jitfunc(arr[0].copy()))