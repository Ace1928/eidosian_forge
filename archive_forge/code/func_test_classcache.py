import unittest
from test.support import bigmemtest, _2G
import sys
from ctypes import *
from ctypes.test import need_symbol
def test_classcache(self):
    self.assertIsNot(ARRAY(c_int, 3), ARRAY(c_int, 4))
    self.assertIs(ARRAY(c_int, 3), ARRAY(c_int, 3))