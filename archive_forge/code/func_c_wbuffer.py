from ctypes import *
from ctypes.test import need_symbol
import unittest
import _ctypes_test
def c_wbuffer(init):
    n = len(init) + 1
    return (c_wchar * n)(*init)