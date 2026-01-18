import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_init_and_equality(self):
    Integer = self.Integer
    v1 = Integer(23)
    v2 = Integer(v1)
    v3 = Integer(-9)
    self.assertRaises(ValueError, Integer, 1.0)
    v4 = Integer(10 ** 10)
    v5 = Integer(-10 ** 10)
    v6 = Integer(65535)
    v7 = Integer(4294967295)
    v8 = Integer(18446744073709551615)
    self.assertEqual(v1, v1)
    self.assertEqual(v1, 23)
    self.assertEqual(v1, v2)
    self.assertEqual(v3, -9)
    self.assertEqual(v4, 10 ** 10)
    self.assertEqual(v5, -10 ** 10)
    self.assertEqual(v6, 65535)
    self.assertEqual(v7, 4294967295)
    self.assertEqual(v8, 18446744073709551615)
    self.assertFalse(v1 == v4)
    v6 = Integer(v1)
    self.assertEqual(v1, v6)
    self.assertFalse(Integer(0) == None)