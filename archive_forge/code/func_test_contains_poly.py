import unittest
import numpy as np
from shapely.geometry import box, MultiPolygon, Point
def test_contains_poly(self):
    y, x = (np.mgrid[-10:10:5j], np.mgrid[-5:15:5j])
    self.assertContainsResults(self.construct_torus(), x, y)