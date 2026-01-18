import unittest
from ctypes import *
def test_1_A(self):

    class X(Structure):
        pass
    self.assertEqual(sizeof(X), 0)
    X._fields_ = []
    self.assertRaises(AttributeError, setattr, X, '_fields_', [])