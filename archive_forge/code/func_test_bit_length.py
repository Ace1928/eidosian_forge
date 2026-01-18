import unittest
import math
import sys
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, tag
def test_bit_length(self):
    f = utils.bit_length
    self.assertEqual(f(127), 7)
    self.assertEqual(f(-127), 7)
    self.assertEqual(f(128), 8)
    self.assertEqual(f(-128), 7)
    self.assertEqual(f(255), 8)
    self.assertEqual(f(-255), 8)
    self.assertEqual(f(256), 9)
    self.assertEqual(f(-256), 8)
    self.assertEqual(f(-257), 9)
    self.assertEqual(f(2147483647), 31)
    self.assertEqual(f(-2147483647), 31)
    self.assertEqual(f(-2147483648), 31)
    self.assertEqual(f(2147483648), 32)
    self.assertEqual(f(4294967295), 32)
    self.assertEqual(f(18446744073709551615), 64)
    self.assertEqual(f(18446744073709551616), 65)