import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
def test_from_linestring_z():
    coords = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    line = LineString(coords)
    copy = LineString(line)
    assert copy.coords[:] == coords
    assert copy.geom_type == 'LineString'