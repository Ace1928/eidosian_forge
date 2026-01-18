import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
def test_linestring_empty(self):
    l_null = LineString()
    assert l_null.wkt == 'LINESTRING EMPTY'
    assert l_null.length == 0.0