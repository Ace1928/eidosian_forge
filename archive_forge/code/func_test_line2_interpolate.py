import unittest
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, Point
def test_line2_interpolate(self):
    assert self.line2.interpolate(0.5).equals(Point(3.0, 0.5))
    assert self.line2.interpolate(0.5, normalized=True).equals(Point(3, 3))