import unittest
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, Point
def test_multiline_project(self):
    assert self.multiline.project(self.point) == 1.0
    assert self.multiline.project(self.point, normalized=True) == 0.125