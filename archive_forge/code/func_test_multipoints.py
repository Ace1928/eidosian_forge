import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
def test_multipoints():
    actual = shapely.multipoints(np.array([point], dtype=object), indices=np.zeros(1, dtype=np.intp))
    assert_geometries_equal(actual, shapely.multipoints([point]))