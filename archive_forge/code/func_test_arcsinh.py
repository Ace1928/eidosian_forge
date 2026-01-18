import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_arcsinh(self):
    q = [1, 2, 3, 4, 6] * pq.dimensionless
    self.assertQuantityEqual(np.arcsinh(q), np.arcsinh(q.magnitude) * pq.rad)