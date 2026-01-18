import unittest
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, Point
def test_line1_project(self):
    assert self.line1.project(self.point) == 1.0
    assert self.line1.project(self.point, normalized=True) == 0.5