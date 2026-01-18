import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_get_bit(self):
    v1, v2, v3 = self.Integers(258, -3, 1)
    self.assertEqual(v1.get_bit(0), 0)
    self.assertEqual(v1.get_bit(1), 1)
    self.assertEqual(v1.get_bit(v3), 1)
    self.assertEqual(v1.get_bit(8), 1)
    self.assertEqual(v1.get_bit(9), 0)
    self.assertRaises(ValueError, v1.get_bit, -1)
    self.assertEqual(v1.get_bit(2 ** 1000), 0)
    self.assertRaises(ValueError, v2.get_bit, -1)
    self.assertRaises(ValueError, v2.get_bit, 0)
    self.assertRaises(ValueError, v2.get_bit, 1)
    self.assertRaises(ValueError, v2.get_bit, 2 * 1000)