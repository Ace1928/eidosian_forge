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
def test_nested_initializers(self):

    class Phone(Structure):
        _fields_ = [('areacode', c_char * 6), ('number', c_char * 12)]

    class Person(Structure):
        _fields_ = [('name', c_char * 12), ('phone', Phone), ('age', c_int)]
    p = Person(b'Someone', (b'1234', b'5678'), 5)
    self.assertEqual(p.name, b'Someone')
    self.assertEqual(p.phone.areacode, b'1234')
    self.assertEqual(p.phone.number, b'5678')
    self.assertEqual(p.age, 5)