import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_no_subtyping2(self):
    jit_fc = njit(self.func)
    jit_fc(self.ab_rec1)
    jit_fc.disable_compile()
    with self.assertRaises(TypeError) as err:
        jit_fc(self.a_rec1)
        self.assertIn('No matching definition for argument type(s) Record', str(err.exception))