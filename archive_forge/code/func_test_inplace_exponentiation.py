import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_inplace_exponentiation(self):
    v1 = self.Integer(4)
    v1.inplace_pow(2)
    self.assertEqual(v1, 16)
    v1 = self.Integer(4)
    v1.inplace_pow(2, 15)
    self.assertEqual(v1, 1)