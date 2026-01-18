import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_conversion_to_bytes(self):
    Integer = self.Integer
    v1 = Integer(23)
    self.assertEqual(b('\x17'), v1.to_bytes())
    v2 = Integer(65534)
    self.assertEqual(b('ÿþ'), v2.to_bytes())
    self.assertEqual(b('\x00ÿþ'), v2.to_bytes(3))
    self.assertRaises(ValueError, v2.to_bytes, 1)
    self.assertEqual(b('þÿ'), v2.to_bytes(byteorder='little'))
    self.assertEqual(b('þÿ\x00'), v2.to_bytes(3, byteorder='little'))
    v3 = Integer(-90)
    self.assertRaises(ValueError, v3.to_bytes)
    self.assertRaises(ValueError, v3.to_bytes, byteorder='bittle')