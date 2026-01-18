import pyomo.common.unittest as unittest
import pytest
from pyomo.contrib.mpc.data.find_nearest_index import (
def test_find_interval(self):
    intervals = [(0.0, 0.1), (0.1, 0.5), (0.7, 1.0)]
    target = 0.05
    idx = find_nearest_interval_index(intervals, target)
    self.assertEqual(idx, 0)
    target = 0.099
    idx = find_nearest_interval_index(intervals, target)
    self.assertEqual(idx, 0)
    target = 0.1
    idx = find_nearest_interval_index(intervals, target)
    self.assertEqual(idx, 0)
    target = 0.1
    idx = find_nearest_interval_index(intervals, target, prefer_left=False)
    self.assertEqual(idx, 1)
    target = 0.55
    idx = find_nearest_interval_index(intervals, target)
    self.assertEqual(idx, 1)
    target = 0.6
    idx = find_nearest_interval_index(intervals, target)
    self.assertEqual(idx, 1)
    target = 0.6999
    idx = find_nearest_interval_index(intervals, target)
    self.assertEqual(idx, 2)
    target = 1.0
    idx = find_nearest_interval_index(intervals, target)
    self.assertEqual(idx, 2)
    target = -0.1
    idx = find_nearest_interval_index(intervals, target)
    self.assertEqual(idx, 0)
    target = 1.1
    idx = find_nearest_interval_index(intervals, target)
    self.assertEqual(idx, 2)