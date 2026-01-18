import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_linearring_from_points():
    expected_coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]
    ring = LinearRing([Point(0.0, 0.0), Point(0.0, 1.0), Point(1.0, 1.0)])
    assert ring.coords[:] == expected_coords