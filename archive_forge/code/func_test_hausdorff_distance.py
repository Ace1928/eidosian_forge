import unittest
import pytest
import shapely
from shapely.errors import TopologicalError
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.wkt import loads
def test_hausdorff_distance(self):
    point = Point(1, 1)
    line = LineString([(2, 0), (2, 4), (3, 4)])
    distance = point.hausdorff_distance(line)
    assert distance == point.distance(Point(3, 4))