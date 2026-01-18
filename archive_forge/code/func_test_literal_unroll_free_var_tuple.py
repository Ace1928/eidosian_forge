import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_literal_unroll_free_var_tuple(self):
    arr = np.array([1, 2], dtype=recordtype2)
    fs = arr.dtype.names

    def set_field(rec):
        for f in literal_unroll(fs):
            rec[f] = 10
        return rec
    jitfunc = njit(set_field)
    self.assertEqual(set_field(arr[0].copy()), jitfunc(arr[0].copy()))