import functools
import unittest
from test import support
from ctypes import *
from ctypes.test import need_symbol
from _ctypes import CTYPES_MAX_ARGCOUNT
import _ctypes_test
@CFUNCTYPE(c_int, POINTER(c_int), POINTER(c_int))
def cmp_func(a, b):
    return a[0] - b[0]