import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.tests.common import all_types, all_types_z, ignore_invalid
def test_equality_exact_type():
    geom1 = LineString([(0, 0), (1, 1), (0, 1), (0, 0)])
    geom2 = LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)])
    geom3 = Polygon([(0, 0), (1, 1), (0, 1), (0, 0)])
    assert geom1 != geom2
    assert geom1 != geom3
    assert geom2 != geom3
    geom1 = shapely.from_wkt('POINT EMPTY')
    geom2 = shapely.from_wkt('LINESTRING EMPTY')
    assert geom1 != geom2