import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_arctan2(self):
    self.assertQuantityEqual(np.arctan2(0 * pq.dimensionless, 0 * pq.dimensionless), 0)
    self.assertQuantityEqual(np.arctan2(3 * pq.V, 3 * pq.V), np.radians(45) * pq.dimensionless)
    self.assertRaises((TypeError, ValueError), np.arctan2, (1 * pq.m, 1 * pq.m))