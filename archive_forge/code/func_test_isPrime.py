import math
import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util import number
from Cryptodome.Util.number import long_to_bytes
def test_isPrime(self):
    """Util.number.isPrime"""
    self.assertEqual(number.isPrime(-3), False)
    self.assertEqual(number.isPrime(-2), False)
    self.assertEqual(number.isPrime(1), False)
    self.assertEqual(number.isPrime(2), True)
    self.assertEqual(number.isPrime(3), True)
    self.assertEqual(number.isPrime(4), False)
    self.assertEqual(number.isPrime(2 ** 1279 - 1), True)
    self.assertEqual(number.isPrime(-(2 ** 1279 - 1)), False)
    for composite in (43 * 127 * 211, 61 * 151 * 211, 15259 * 30517, 346141 * 692281, 1007119 * 2014237, 3589477 * 7178953, 4859419 * 9718837, 2730439 * 5460877, 245127919 * 490255837, 963939391 * 1927878781, 4186358431 * 8372716861, 1576820467 * 3153640933):
        self.assertEqual(number.isPrime(int(composite)), False)