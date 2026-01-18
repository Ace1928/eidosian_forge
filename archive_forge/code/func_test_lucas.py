import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (
def test_lucas(self):
    for prime in self.primes:
        res = lucas_test(prime)
        self.assertEqual(res, PROBABLY_PRIME)
    for composite in self.composites:
        res = lucas_test(composite)
        self.assertEqual(res, COMPOSITE)
    self.assertRaises(ValueError, lucas_test, -1)