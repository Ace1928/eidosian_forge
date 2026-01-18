import unittest
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, Point
def test_multiline_interpolate(self):
    assert self.multiline.interpolate(0.5).equals(Point(0.5, 0))
    assert self.multiline.interpolate(0.5, normalized=True).equals(Point(3.0, 2.0))