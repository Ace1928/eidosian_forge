import pyomo.common.unittest as unittest
import pytest
from pyomo.contrib.mpc.data.find_nearest_index import (
def test_array_with_floats(self):
    array = []
    for i in range(5):
        i0 = float(i)
        i1 = round((i + 0.15) * 10000.0) / 10000.0
        i2 = round((i + 0.64) * 10000.0) / 10000.0
        array.extend([i, i1, i2])
    array.append(5.0)
    i = find_nearest_index(array, 1.01, tolerance=0.1)
    self.assertEqual(i, 3)
    i = find_nearest_index(array, 1.01, tolerance=0.001)
    self.assertEqual(i, None)
    i = find_nearest_index(array, 3.5)
    self.assertEqual(i, 11)
    i = find_nearest_index(array, 3.5, tolerance=0.1)
    self.assertEqual(i, None)
    i = find_nearest_index(array, -1)
    self.assertEqual(i, 0)
    i = find_nearest_index(array, -1, tolerance=1)
    self.assertEqual(i, 0)
    i = find_nearest_index(array, 5.5)
    self.assertEqual(i, 15)
    i = find_nearest_index(array, 5.5, tolerance=0.49)
    self.assertEqual(i, None)
    i = find_nearest_index(array, 2.64, tolerance=1e-08)
    self.assertEqual(i, 8)
    i = find_nearest_index(array, 2.64, tolerance=0)
    self.assertEqual(i, 8)
    i = find_nearest_index(array, 5, tolerance=0)
    self.assertEqual(i, 15)
    i = find_nearest_index(array, 0, tolerance=0)
    self.assertEqual(i, 0)