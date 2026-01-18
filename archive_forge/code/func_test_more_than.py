import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_more_than(self):
    v1, v2, v3, v4, v5 = self.Integers(13, 13, 14, -8, 2 ** 10)
    self.assertTrue(v3 > v1)
    self.assertTrue(v3 > 13)
    self.assertFalse(v1 > v1)
    self.assertFalse(v1 > v2)
    self.assertFalse(v1 > 13)
    self.assertTrue(v1 > v4)
    self.assertFalse(v4 > v1)
    self.assertTrue(v5 > v1)
    self.assertFalse(v1 > v5)