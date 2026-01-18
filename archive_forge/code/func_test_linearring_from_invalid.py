import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_linearring_from_invalid():
    coords = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    line = LineString(coords)
    assert not line.is_valid
    with pytest.raises(TopologicalError):
        LinearRing(line)