import unittest
import pytest
from shapely.algorithms.polylabel import Cell, polylabel
from shapely.errors import TopologicalError
from shapely.geometry import LineString, Point, Polygon
def test_concave_polygon(self):
    """
        Finds pole of inaccessibility for a concave polygon and ensures that
        the point is inside.

        """
    concave_polygon = LineString([(500, 0), (0, 0), (0, 500), (500, 500)]).buffer(100)
    label = polylabel(concave_polygon)
    assert concave_polygon.contains(label)