import operator as op
from .. import units as pq
from ..dimensionality import Dimensionality
from .common import TestCase
def test_inplace_division(self):
    temp = meter.copy()
    temp /= meter
    self.assertEqual(temp, meter / meter)
    temp /= centimeter
    self.assertEqual(temp, meter / meter / centimeter)
    temp /= centimeter ** (-1)
    self.assertEqual(temp, meter / meter)
    self.assertRaises(TypeError, op.itruediv, Joule, 0)
    self.assertRaises(TypeError, op.itruediv, 0, joule)