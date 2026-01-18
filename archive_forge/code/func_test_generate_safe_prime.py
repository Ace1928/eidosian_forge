import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (
def test_generate_safe_prime(self):
    p = generate_probable_safe_prime(exact_bits=161)
    self.assertEqual(p.size_in_bits(), 161)