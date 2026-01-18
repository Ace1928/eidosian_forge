from ctypes import *
import unittest, sys
from test import support
import _ctypes_test
def test_COMError(self):
    from _ctypes import COMError
    if support.HAVE_DOCSTRINGS:
        self.assertEqual(COMError.__doc__, 'Raised when a COM method call failed.')
    ex = COMError(-1, 'text', ('details',))
    self.assertEqual(ex.hresult, -1)
    self.assertEqual(ex.text, 'text')
    self.assertEqual(ex.details, ('details',))