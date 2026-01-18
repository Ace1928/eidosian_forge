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
def test_conflicting_initializers(self):

    class POINT(Structure):
        _fields_ = [('phi', c_float), ('rho', c_float)]
    self.assertRaisesRegex(TypeError, 'phi', POINT, 2, 3, phi=4)
    self.assertRaisesRegex(TypeError, 'rho', POINT, 2, 3, rho=4)
    self.assertRaises(TypeError, POINT, 2, 3, 4)