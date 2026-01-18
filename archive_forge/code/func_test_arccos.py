import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_arccos(self):
    self.assertQuantityEqual(np.arccos(1 * pq.dimensionless), 0 * pq.radian)
    self.assertRaises(ValueError, np.arccos, 1 * pq.m)