from ctypes import *
import unittest, sys
from test import support
import _ctypes_test
def test_PARAM(self):
    from ctypes import wintypes
    self.assertEqual(sizeof(wintypes.WPARAM), sizeof(c_void_p))
    self.assertEqual(sizeof(wintypes.LPARAM), sizeof(c_void_p))