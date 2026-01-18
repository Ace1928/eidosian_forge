import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_arctan(self):
    self.assertQuantityEqual(np.arctan(0 * pq.dimensionless), 0 * pq.radian)
    self.assertRaises(ValueError, np.arctan, 1 * pq.m)