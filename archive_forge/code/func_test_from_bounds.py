import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_from_bounds(self):
    xmin, ymin, xmax, ymax = (-180, -90, 180, 90)
    coords = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
    assert Polygon(coords) == Polygon.from_bounds(xmin, ymin, xmax, ymax)