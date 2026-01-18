import logging
import math
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import (
from pyomo.environ import (
import pyomo.repn.util
from pyomo.repn.util import (
def test_InvalidNumber(self):
    a = InvalidNumber(-3)
    b = InvalidNumber(5)
    c = InvalidNumber(5)
    self.assertEqual((a + b).value, 2)
    self.assertEqual((a - b).value, -8)
    self.assertEqual((a * b).value, -15)
    self.assertEqual((a / b).value, -0.6)
    self.assertEqual((a ** b).value, -3 ** 5)
    self.assertEqual(abs(a).value, 3)
    self.assertEqual(abs(b).value, 5)
    self.assertEqual((-a).value, 3)
    self.assertEqual((-b).value, -5)
    self.assertEqual((a + 5).value, 2)
    self.assertEqual((a - 5).value, -8)
    self.assertEqual((a * 5).value, -15)
    self.assertEqual((a / 5).value, -0.6)
    self.assertEqual((a ** 5).value, -3 ** 5)
    self.assertEqual((-3 + b).value, 2)
    self.assertEqual((-3 - b).value, -8)
    self.assertEqual((-3 * b).value, -15)
    self.assertEqual((-3 / b).value, -0.6)
    self.assertEqual(((-3) ** b).value, -3 ** 5)
    self.assertTrue(a < b)
    self.assertTrue(a <= b)
    self.assertFalse(a > b)
    self.assertFalse(a >= b)
    self.assertFalse(a == b)
    self.assertTrue(a != b)
    self.assertFalse(c < b)
    self.assertTrue(c <= b)
    self.assertFalse(c > b)
    self.assertTrue(c >= b)
    self.assertTrue(c == b)
    self.assertFalse(c != b)
    self.assertTrue(a < 5)
    self.assertTrue(a <= 5)
    self.assertFalse(a > 5)
    self.assertFalse(a >= 5)
    self.assertFalse(a == 5)
    self.assertTrue(a != 5)
    self.assertTrue(3 < b)
    self.assertTrue(3 <= b)
    self.assertFalse(3 > b)
    self.assertFalse(3 >= b)
    self.assertFalse(3 == b)
    self.assertTrue(3 != b)
    d = InvalidNumber('abc')
    with self.assertRaisesRegex(InvalidValueError, 'Cannot emit InvalidNumber\\(5\\) in compiled representation'):
        repr(b)
    with self.assertRaisesRegex(InvalidValueError, "Cannot emit InvalidNumber\\('abc'\\) in compiled representation"):
        repr(d)
    with self.assertRaisesRegex(InvalidValueError, 'Cannot emit InvalidNumber\\(5\\) in compiled representation'):
        f'{b}'
    with self.assertRaisesRegex(InvalidValueError, "Cannot emit InvalidNumber\\('abc'\\) in compiled representation"):
        f'{d}'