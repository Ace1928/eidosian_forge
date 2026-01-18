import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_fail_if_divisible_by(self):
    v1, v2, v3 = self.Integers(12, -12, 4)
    v1.fail_if_divisible_by(7)
    v2.fail_if_divisible_by(7)
    v2.fail_if_divisible_by(2 ** 80)
    self.assertRaises(ValueError, v1.fail_if_divisible_by, 4)
    self.assertRaises(ValueError, v1.fail_if_divisible_by, v3)