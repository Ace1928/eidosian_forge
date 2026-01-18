import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_modular_exponentiation(self):
    v1, v2, v3 = self.Integers(23, 5, 17)
    self.assertTrue(isinstance(pow(v1, v2, v3), self.Integer))
    self.assertEqual(pow(v1, v2, v3), 7)
    self.assertEqual(pow(v1, 5, v3), 7)
    self.assertEqual(pow(v1, v2, 17), 7)
    self.assertEqual(pow(v1, 5, 17), 7)
    self.assertEqual(pow(v1, 0, 17), 1)
    self.assertEqual(pow(v1, 1, 2 ** 80), 23)
    self.assertEqual(pow(v1, 2 ** 80, 89298), 17689)
    self.assertRaises(ZeroDivisionError, pow, v1, 5, 0)
    self.assertRaises(ValueError, pow, v1, 5, -4)
    self.assertRaises(ValueError, pow, v1, -3, 8)