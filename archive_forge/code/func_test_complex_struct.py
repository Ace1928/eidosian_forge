import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_complex_struct(self):

    class Complex(ctypes.Structure):
        _fields_ = [('real', ctypes.c_double), ('imag', ctypes.c_double)]

    class Ref(ctypes.Structure):
        _fields_ = [('apple', ctypes.c_int32), ('mango', Complex)]
    ty = types.Record.make_c_struct([('apple', types.intc), ('mango', types.complex128)])
    self.assertEqual(len(ty), 2)
    self.assertEqual(ty.offset('apple'), Ref.apple.offset)
    self.assertEqual(ty.offset('mango'), Ref.mango.offset)
    self.assertEqual(ty.size, ctypes.sizeof(Ref))
    self.assertTrue(ty.dtype.isalignedstruct)