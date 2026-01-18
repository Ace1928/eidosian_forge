import operator as op
from .. import units as pq
from ..dimensionality import Dimensionality
from .common import TestCase
def test_inplace_addition(self):
    temp = meter.copy()
    temp += meter
    self.assertEqual(temp, meter)
    self.assertRaises(ValueError, op.iadd, meter, joule)
    self.assertRaises(ValueError, op.iadd, Joule, joule)
    self.assertRaises(TypeError, op.iadd, Joule, 0)
    self.assertRaises(TypeError, op.iadd, 0, joule)