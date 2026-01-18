import math
import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util import number
from Cryptodome.Util.number import long_to_bytes
def test_getStrongPrime(self):
    """Util.number.getStrongPrime"""
    self.assertRaises(ValueError, number.getStrongPrime, 256)
    self.assertRaises(ValueError, number.getStrongPrime, 513)
    bits = 512
    x = number.getStrongPrime(bits)
    self.assertNotEqual(x % 2, 0)
    self.assertEqual(x > (1 << bits - 1) - 1, 1)
    self.assertEqual(x < 1 << bits, 1)
    e = 2 ** 16 + 1
    x = number.getStrongPrime(bits, e)
    self.assertEqual(number.GCD(x - 1, e), 1)
    self.assertNotEqual(x % 2, 0)
    self.assertEqual(x > (1 << bits - 1) - 1, 1)
    self.assertEqual(x < 1 << bits, 1)
    e = 2 ** 16 + 2
    x = number.getStrongPrime(bits, e)
    self.assertEqual(number.GCD(x - 1 >> 1, e), 1)
    self.assertNotEqual(x % 2, 0)
    self.assertEqual(x > (1 << bits - 1) - 1, 1)
    self.assertEqual(x < 1 << bits, 1)