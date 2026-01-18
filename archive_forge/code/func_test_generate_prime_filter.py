import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (
def test_generate_prime_filter(self):

    def ending_with_one(number):
        return number % 10 == 1
    for x in range(20):
        q = generate_probable_prime(exact_bits=160, prime_filter=ending_with_one)
        self.assertEqual(q % 10, 1)