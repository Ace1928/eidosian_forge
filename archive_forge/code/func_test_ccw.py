import unittest
import pytest
from shapely.geometry.polygon import LinearRing, orient, Polygon, signed_area
def test_ccw(self):
    ring = LinearRing([(1, 0), (0, 1), (0, 0)])
    assert ring.is_ccw