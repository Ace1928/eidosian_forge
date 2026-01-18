import unittest
from test.support import bigmemtest, _2G
import sys
from ctypes import *
from ctypes.test import need_symbol
def test_bpo36504_signed_int_overflow(self):
    with self.assertRaises(OverflowError):
        c_char * sys.maxsize * 2