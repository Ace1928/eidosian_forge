import unittest
from test.support import bigmemtest, _2G
import sys
from ctypes import *
from ctypes.test import need_symbol
def test_empty_element_array(self):

    class EmptyArray(Array):
        _type_ = c_int
        _length_ = 0
    obj = (EmptyArray * 2)()
    self.assertEqual(sizeof(obj), 0)