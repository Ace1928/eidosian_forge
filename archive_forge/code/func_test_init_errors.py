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
def test_init_errors(self):

    class Phone(Structure):
        _fields_ = [('areacode', c_char * 6), ('number', c_char * 12)]

    class Person(Structure):
        _fields_ = [('name', c_char * 12), ('phone', Phone), ('age', c_int)]
    cls, msg = self.get_except(Person, b'Someone', (1, 2))
    self.assertEqual(cls, RuntimeError)
    self.assertEqual(msg, '(Phone) TypeError: expected bytes, int found')
    cls, msg = self.get_except(Person, b'Someone', (b'a', b'b', b'c'))
    self.assertEqual(cls, RuntimeError)
    self.assertEqual(msg, '(Phone) TypeError: too many initializers')