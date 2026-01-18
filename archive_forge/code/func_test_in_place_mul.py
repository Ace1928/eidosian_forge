import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_in_place_mul(self):
    v1, v2 = self.Integers(3, 5)
    v1 *= v2
    self.assertEqual(v1, 15)
    v1 *= 2
    self.assertEqual(v1, 30)
    v1 *= -2
    self.assertEqual(v1, -60)
    v1 *= 2 ** 1000
    self.assertEqual(v1, -60 * 2 ** 1000)