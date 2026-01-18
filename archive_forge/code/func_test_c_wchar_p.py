import unittest
import sys
from ctypes import *
def test_c_wchar_p(self):
    c_wchar_p('foo bar')
    self.assertRaises(TypeError, c_wchar_p, b'foo bar')