import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_cumprod(self):
    self.assertRaises(ValueError, np.cumprod, self.q)
    q = [10, 0.1, 5, 50] * pq.dimensionless
    self.assertQuantityEqual(np.cumprod(q), [10, 1, 5, 250])