import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_in_place_left_shift(self):
    v1, v2, v3 = self.Integers(16, 1, -16)
    v1 <<= 0
    self.assertEqual(v1, 16)
    v1 <<= 1
    self.assertEqual(v1, 32)
    v1 <<= v2
    self.assertEqual(v1, 64)
    v3 <<= 1
    self.assertEqual(v3, -32)

    def l():
        v4 = self.Integer(144)
        v4 <<= -1
    self.assertRaises(ValueError, l)

    def m():
        v4 = self.Integer(144)
        v4 <<= 2 ** 1000
    self.assertRaises(ValueError, m)