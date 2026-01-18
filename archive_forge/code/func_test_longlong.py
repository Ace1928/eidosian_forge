import functools
import unittest
from test import support
from ctypes import *
from ctypes.test import need_symbol
from _ctypes import CTYPES_MAX_ARGCOUNT
import _ctypes_test
@need_symbol('c_longlong')
def test_longlong(self):
    self.check_type(c_longlong, 42)
    self.check_type(c_longlong, -42)