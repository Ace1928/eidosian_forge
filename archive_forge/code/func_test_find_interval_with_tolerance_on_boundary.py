import pyomo.common.unittest as unittest
import pytest
from pyomo.contrib.mpc.data.find_nearest_index import (
def test_find_interval_with_tolerance_on_boundary(self):
    intervals = [(0.0, 0.1), (0.1, 0.5), (0.5, 1.0)]
    target = 0.1001
    idx = find_nearest_interval_index(intervals, target, tolerance=None, prefer_left=True)
    self.assertEqual(idx, 1)
    idx = find_nearest_interval_index(intervals, target, tolerance=1e-05, prefer_left=True)
    self.assertEqual(idx, 1)
    idx = find_nearest_interval_index(intervals, target, tolerance=1e-05, prefer_left=False)
    self.assertEqual(idx, 1)
    idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=True)
    self.assertEqual(idx, 0)
    idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=False)
    self.assertEqual(idx, 1)
    target = 0.4999
    idx = find_nearest_interval_index(intervals, target, tolerance=1e-05, prefer_left=True)
    self.assertEqual(idx, 1)
    idx = find_nearest_interval_index(intervals, target, tolerance=1e-05, prefer_left=False)
    self.assertEqual(idx, 1)
    idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=True)
    self.assertEqual(idx, 1)
    idx = find_nearest_interval_index(intervals, target, tolerance=0.001, prefer_left=False)
    self.assertEqual(idx, 2)