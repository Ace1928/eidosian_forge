import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_ediff1d(self):
    self.assertQuantityEqual(np.diff(self.q, 1), [1, 1, 1] * pq.J)