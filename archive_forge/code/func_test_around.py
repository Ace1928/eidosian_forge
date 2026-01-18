import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_around(self):
    self.assertQuantityEqual(np.around([0.5, 1.5, 2.5, 3.5, 4.5] * pq.J), [0.0, 2.0, 2.0, 4.0, 4.0] * pq.J)
    self.assertQuantityEqual(np.around([1, 2, 3, 11] * pq.J, decimals=1), [1, 2, 3, 11] * pq.J)
    self.assertQuantityEqual(np.around([1, 2, 3, 11] * pq.J, decimals=-1), [0, 0, 0, 10] * pq.J)