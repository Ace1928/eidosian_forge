import platform
from platform import architecture as _architecture
import struct
import sys
import unittest
from ctypes.test import need_symbol
from ctypes import (CDLL, Array, Structure, Union, POINTER, sizeof, byref, alignment,
from ctypes.util import find_library
from struct import calcsize
import _ctypes_test
from collections import namedtuple
from test import support
def test_38368(self):

    class U(Union):
        _fields_ = [('f1', c_uint8 * 16), ('f2', c_uint16 * 8), ('f3', c_uint32 * 4)]
    u = U()
    u.f3[0] = 19088743
    u.f3[1] = 2309737967
    u.f3[2] = 1985229328
    u.f3[3] = 4275878552
    f1 = [u.f1[i] for i in range(16)]
    f2 = [u.f2[i] for i in range(8)]
    if sys.byteorder == 'little':
        self.assertEqual(f1, [103, 69, 35, 1, 239, 205, 171, 137, 16, 50, 84, 118, 152, 186, 220, 254])
        self.assertEqual(f2, [17767, 291, 52719, 35243, 12816, 30292, 47768, 65244])