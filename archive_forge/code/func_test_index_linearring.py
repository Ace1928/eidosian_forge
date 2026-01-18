import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_index_linearring(self):
    shell = LinearRing([(0.0, 0.0), (70.0, 120.0), (140.0, 0.0), (0.0, 0.0)])
    holes = [LinearRing([(60.0, 80.0), (80.0, 80.0), (70.0, 60.0), (60.0, 80.0)]), LinearRing([(30.0, 10.0), (50.0, 10.0), (40.0, 30.0), (30.0, 10.0)]), LinearRing([(90.0, 10), (110.0, 10.0), (100.0, 30.0), (90.0, 10.0)])]
    g = Polygon(shell, holes)
    for i in range(-3, 3):
        assert g.interiors[i].equals(holes[i])
    with pytest.raises(IndexError):
        g.interiors[3]
    with pytest.raises(IndexError):
        g.interiors[-4]