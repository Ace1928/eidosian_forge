import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_in_place_modulus(self):
    v1, v2 = self.Integers(20, 7)
    v1 %= v2
    self.assertEqual(v1, 6)
    v1 %= 2 ** 1000
    self.assertEqual(v1, 6)
    v1 %= 2
    self.assertEqual(v1, 0)

    def t():
        v3 = self.Integer(9)
        v3 %= 0
    self.assertRaises(ZeroDivisionError, t)