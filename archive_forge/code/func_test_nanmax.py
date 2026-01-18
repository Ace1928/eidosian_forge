from .. import units as pq
from .common import TestCase
import numpy as np
def test_nanmax(self):
    q = np.append(self.q, np.nan) * self.q.units
    self.assertQuantityEqual(q.nanmax(), 4 * pq.m)