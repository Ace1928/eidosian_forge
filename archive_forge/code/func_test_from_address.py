import unittest, sys
from ctypes import *
import _ctypes_test
def test_from_address(self):
    from array import array
    a = array('i', [100, 200, 300, 400, 500])
    addr = a.buffer_info()[0]
    p = POINTER(POINTER(c_int))