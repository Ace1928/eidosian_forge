import numpy as np
from numba.core.errors import TypingError
from numba import njit
from numba.core import types
import struct
import unittest
def test_float_to_unsigned(self):
    pyfunc = float_to_unsigned
    cfunc = njit((types.float32,))(pyfunc)
    self.assertEqual(cfunc.nopython_signatures[0].return_type, types.uint32)
    self.assertEqual(cfunc(3.21), pyfunc(3.21))
    self.assertEqual(cfunc(3.21), struct.unpack('I', struct.pack('i', 3))[0])