import functools
import unittest
from test import support
from ctypes import *
from ctypes.test import need_symbol
from _ctypes import CTYPES_MAX_ARGCOUNT
import _ctypes_test
def test_double(self):
    self.check_type(c_double, 3.14)
    self.check_type(c_double, -3.14)