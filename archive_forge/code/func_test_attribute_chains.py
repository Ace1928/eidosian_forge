import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_attribute_chains(self):
    p = Polygon([(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)])
    assert list(p.boundary.coords) == [(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 0.0)]
    ec = list(Point(0.0, 0.0).buffer(1.0, 1).exterior.coords)
    assert isinstance(ec, list)
    p = Polygon([(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)], [[(-0.25, 0.25), (-0.25, 0.75), (-0.75, 0.75), (-0.75, 0.25)]])
    assert p.area == 0.75
    'Not so much testing the exact values here, which are the\n        responsibility of the geometry engine (GEOS), but that we can get\n        chain functions and properties using anonymous references.\n        '
    assert list(p.interiors[0].coords) == [(-0.25, 0.25), (-0.25, 0.75), (-0.75, 0.75), (-0.75, 0.25), (-0.25, 0.25)]
    xy = list(p.interiors[0].buffer(1).exterior.coords)[0]
    assert len(xy) == 2
    ec = list(p.buffer(1).boundary.coords)
    assert isinstance(ec, list)