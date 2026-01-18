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
def test_initializers(self):

    class Person(Structure):
        _fields_ = [('name', c_char * 6), ('age', c_int)]
    self.assertRaises(TypeError, Person, 42)
    self.assertRaises(ValueError, Person, b'asldkjaslkdjaslkdj')
    self.assertRaises(TypeError, Person, 'Name', 'HI')
    self.assertEqual(Person(b'12345', 5).name, b'12345')
    self.assertEqual(Person(b'123456', 5).name, b'123456')
    self.assertRaises(ValueError, Person, b'1234567', 5)