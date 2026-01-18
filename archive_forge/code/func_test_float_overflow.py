from ctypes import *
import unittest
import struct
def test_float_overflow(self):
    import sys
    big_int = int(sys.float_info.max) * 2
    for t in float_types + [c_longdouble]:
        self.assertRaises(OverflowError, t, big_int)
        if hasattr(t, '__ctype_be__'):
            self.assertRaises(OverflowError, t.__ctype_be__, big_int)
        if hasattr(t, '__ctype_le__'):
            self.assertRaises(OverflowError, t.__ctype_le__, big_int)