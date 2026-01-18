import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_floor_div(self):
    v1, v2, v3 = self.Integers(3, 8, 2 ** 80)
    self.assertTrue(isinstance(v1 // v2, self.Integer))
    self.assertEqual(v2 // v1, 2)
    self.assertEqual(v2 // 3, 2)
    self.assertEqual(v2 // -3, -3)
    self.assertEqual(v3 // 2 ** 79, 2)
    self.assertRaises(ZeroDivisionError, lambda: v1 // 0)