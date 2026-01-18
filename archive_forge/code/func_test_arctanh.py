import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_arctanh(self):
    q = [0.01, 0.5, 0.6, 0.8, 0.99] * pq.dimensionless
    self.assertQuantityEqual(np.arctanh(q), np.arctanh(q.magnitude) * pq.rad)