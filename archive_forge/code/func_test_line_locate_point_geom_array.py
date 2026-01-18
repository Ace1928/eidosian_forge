import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_line_locate_point_geom_array():
    point = shapely.points(0, 1)
    actual = shapely.line_locate_point([line_string, linear_ring], point)
    np.testing.assert_allclose(actual, [0.0, 3.0])