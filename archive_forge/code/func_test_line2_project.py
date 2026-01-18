import unittest
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, Point
def test_line2_project(self):
    assert self.line2.project(self.point) == 1.0
    assert self.line2.project(self.point, normalized=True) == pytest.approx(0.16666666666, 8)