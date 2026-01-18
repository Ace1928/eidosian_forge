import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
def test_from_coordinate_sequence_3D():
    line = LineString([(1.0, 2.0, 3.0), (3.0, 4.0, 5.0)])
    assert line.has_z
    assert line.coords[:] == [(1.0, 2.0, 3.0), (3.0, 4.0, 5.0)]