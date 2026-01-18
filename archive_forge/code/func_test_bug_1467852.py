import unittest, sys
from ctypes import *
import _ctypes_test
def test_bug_1467852(self):
    x = c_int(5)
    dummy = []
    for i in range(32000):
        dummy.append(c_int(i))
    y = c_int(6)
    p = pointer(x)
    pp = pointer(p)
    q = pointer(y)
    pp[0] = q
    self.assertEqual(p[0], 6)