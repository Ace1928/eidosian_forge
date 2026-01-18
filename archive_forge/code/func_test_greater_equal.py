import numpy as np
from .. import units as pq
from .common import TestCase, unittest
def test_greater_equal(self):
    arr1 = (1, 1) * pq.m
    arr2 = (1.0, 2.0) * pq.m
    self.assertTrue(np.all(np.greater_equal(arr2, arr1)))
    self.assertFalse(np.all(np.greater_equal(arr2 * 0.99, arr1)))