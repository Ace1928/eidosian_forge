import pyomo.common.unittest as unittest
import pytest
from pyomo.contrib.mpc.data.find_nearest_index import (
def test_find_interval_tolerance(self):
    intervals = [(0.0, 0.1), (0.1, 0.5), (0.7, 1.0)]
    target = 0.501
    idx = find_nearest_interval_index(intervals, target, tolerance=None)
    self.assertEqual(idx, 1)
    idx = find_nearest_interval_index(intervals, target, tolerance=1e-05)
    self.assertEqual(idx, None)
    idx = find_nearest_interval_index(intervals, target, tolerance=0.01)
    self.assertEqual(idx, 1)
    target = 1.001
    idx = find_nearest_interval_index(intervals, target, tolerance=0.01)
    self.assertEqual(idx, 2)
    idx = find_nearest_interval_index(intervals, target, tolerance=0.0001)
    self.assertEqual(idx, None)