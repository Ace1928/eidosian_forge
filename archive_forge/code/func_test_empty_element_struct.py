import unittest
from test.support import bigmemtest, _2G
import sys
from ctypes import *
from ctypes.test import need_symbol
def test_empty_element_struct(self):

    class EmptyStruct(Structure):
        _fields_ = []
    obj = (EmptyStruct * 2)()
    self.assertEqual(sizeof(obj), 0)