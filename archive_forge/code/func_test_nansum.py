import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_nansum(self):
    c = [1, 2, 3, np.nan] * pq.m
    self.assertQuantityEqual(np.nansum(c), 6 * pq.m)