import operator as op
from .. import units as pq
from ..dimensionality import Dimensionality
from .common import TestCase
def test_inplace_subtraction(self):
    temp = meter.copy()
    temp -= meter
    self.assertEqual(temp, meter)
    self.assertRaises(ValueError, op.isub, meter, joule)
    self.assertRaises(ValueError, op.isub, Joule, joule)
    self.assertRaises(TypeError, op.isub, Joule, 0)
    self.assertRaises(TypeError, op.isub, 0, joule)