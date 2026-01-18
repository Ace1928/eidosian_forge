from ctypes import *
from ctypes.test import need_symbol
import unittest
import sys
def test_char_p(self):
    s = c_char_p(b'hiho')
    self.assertEqual(cast(cast(s, c_void_p), c_char_p).value, b'hiho')