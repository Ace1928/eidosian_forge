import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_line_interpolate_point_geom_array_normalized():
    actual = shapely.line_interpolate_point([line_string, linear_ring, multi_line_string], 1, normalized=True)
    assert_geometries_equal(actual[0], Point(1, 1))
    assert_geometries_equal(actual[1], Point(0, 0))
    assert_geometries_equal(actual[2], Point(1, 2))