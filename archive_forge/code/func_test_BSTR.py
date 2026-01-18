import unittest
import sys
from ctypes import *
@unittest.skipUnless(sys.platform == 'win32', 'Windows-specific test')
def test_BSTR(self):
    from _ctypes import _SimpleCData

    class BSTR(_SimpleCData):
        _type_ = 'X'
    BSTR('abc')