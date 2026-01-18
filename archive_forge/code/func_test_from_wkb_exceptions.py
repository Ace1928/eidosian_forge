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
def test_from_wkb_exceptions():
    with pytest.raises(TypeError, match='Expected bytes or string, got int'):
        shapely.from_wkb(1)
    with pytest.raises(shapely.GEOSException, match='Unexpected EOF parsing WKB|ParseException: Input buffer is smaller than requested object size'):
        result = shapely.from_wkb(b'\x01\x01\x00\x00\x00\x00')
        assert result is None
    with pytest.raises(shapely.GEOSException, match='Points of LinearRing do not form a closed linestring'):
        result = shapely.from_wkb(INVALID_WKB)
        assert result is None