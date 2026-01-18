import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('func', [shapely.multipoints, shapely.multilinestrings, shapely.multipolygons, shapely.geometrycollections])
@pytest.mark.parametrize('geometries', [np.array([1, 2], dtype=np.intp), None, np.array([[point]]), 'hello'])
def test_invalid_geometries(func, geometries):
    with pytest.raises((TypeError, ValueError)):
        func(geometries, indices=[0, 1])