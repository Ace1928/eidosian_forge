import json
import pickle
import struct
import warnings
import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LineString, Point, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types, empty_point, empty_point_z, point, point_z
@pytest.mark.parametrize('wkt', ('POINT EMPTY', 'LINESTRING EMPTY', 'POLYGON EMPTY', 'GEOMETRYCOLLECTION EMPTY'))
def test_from_wkt_empty(wkt):
    geom = shapely.from_wkt(wkt)
    assert shapely.is_geometry(geom).all()
    assert shapely.is_empty(geom).all()
    assert shapely.to_wkt(geom) == wkt