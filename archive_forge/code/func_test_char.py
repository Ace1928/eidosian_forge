import functools
import unittest
from test import support
from ctypes import *
from ctypes.test import need_symbol
from _ctypes import CTYPES_MAX_ARGCOUNT
import _ctypes_test
def test_char(self):
    self.check_type(c_char, b'x')
    self.check_type(c_char, b'a')