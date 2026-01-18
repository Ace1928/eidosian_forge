import operator as op
from .. import units as pq
from ..dimensionality import Dimensionality
from .common import TestCase
def test_inplace_power(self):
    temp = meter.copy()
    temp **= 2
    self.assertEqual(temp, meter ** 2)
    temp = joule.copy()
    temp **= 2
    self.assertEqual(temp, joule ** 2)
    temp = meter.copy()
    temp **= 0
    self.assertEqual(temp, Dimensionality())
    self.assertRaises(TypeError, op.ipow, Joule, joule)
    self.assertRaises(TypeError, op.ipow, joule, Joule)