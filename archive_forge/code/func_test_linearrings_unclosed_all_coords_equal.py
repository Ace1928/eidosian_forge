import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
def test_linearrings_unclosed_all_coords_equal():
    actual = shapely.linearrings([(0, 0), (0, 0), (0, 0)], indices=np.zeros(3))
    assert_geometries_equal(actual, LinearRing([(0, 0), (0, 0), (0, 0), (0, 0)]))