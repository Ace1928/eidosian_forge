import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.tests.common import all_types, all_types_z, ignore_invalid
def test_equality_polygon():
    geom1 = shapely.from_wkt('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))')
    geom2 = shapely.from_wkt('POLYGON ((0 0, 10 0, 10 10, 0 15, 0 0))')
    assert geom1 != geom2
    geom1 = shapely.from_wkt('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1))')
    geom2 = shapely.from_wkt('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1), (3 3, 4 3, 4 4, 3 3))')
    assert geom1 != geom2
    geom1 = shapely.from_wkt('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 4 3, 4 4, 3 3), (1 1, 2 1, 2 2, 1 1))')
    geom2 = shapely.from_wkt('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1), (3 3, 4 3, 4 4, 3 3))')
    assert geom1 != geom2