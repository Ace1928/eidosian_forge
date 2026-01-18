import sys
import ctypes
from ctypes import *
import unittest
def test__addr_at(self):
    a = self.a
    self.assertEqual(a._addr_at((0, 0)), a.data)
    self.assertEqual(a._addr_at((0, 1)), a.data + 4)
    self.assertEqual(a._addr_at((1, 0)), a.data + 60)
    self.assertEqual(a._addr_at((1, 1)), a.data + 64)