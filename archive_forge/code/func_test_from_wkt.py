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
def test_from_wkt():
    expected = shapely.points(1, 1)
    actual = shapely.from_wkt('POINT (1 1)')
    assert_geometries_equal(actual, expected)
    actual = shapely.from_wkt(b'POINT (1 1)')
    assert_geometries_equal(actual, expected)