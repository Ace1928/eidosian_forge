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
def test_abstract_class(self):

    class X(Structure):
        _abstract_ = 'something'
    cls, msg = self.get_except(eval, 'X()', locals())
    self.assertEqual((cls, msg), (TypeError, 'abstract class'))