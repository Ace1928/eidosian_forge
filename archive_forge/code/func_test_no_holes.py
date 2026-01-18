import unittest
import pytest
from shapely.geometry.polygon import LinearRing, orient, Polygon, signed_area
def test_no_holes(self):
    ring = LinearRing([(0, 0), (0, 1), (1, 0)])
    polygon = Polygon(ring)
    assert not polygon.exterior.is_ccw
    polygon = orient(polygon, 1)
    assert polygon.exterior.is_ccw