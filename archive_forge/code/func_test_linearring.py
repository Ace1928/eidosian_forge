import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_linearring(self):
    coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    ring = LinearRing(coords)
    assert len(ring.coords) == 5
    assert ring.coords[0] == ring.coords[4]
    assert ring.coords[0] == ring.coords[-1]
    assert ring.is_ring is True