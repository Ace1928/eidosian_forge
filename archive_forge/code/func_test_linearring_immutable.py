import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_linearring_immutable():
    ring = LinearRing([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
    with pytest.raises(AttributeError):
        ring.coords = [(1.0, 1.0), (2.0, 2.0), (1.0, 2.0)]
    with pytest.raises(TypeError):
        ring.coords[0] = (1.0, 1.0)