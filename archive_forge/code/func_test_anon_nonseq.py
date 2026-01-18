import unittest
import test.support
from ctypes import *
def test_anon_nonseq(self):
    self.assertRaises(TypeError, lambda: type(Structure)('Name', (Structure,), {'_fields_': [], '_anonymous_': 42}))