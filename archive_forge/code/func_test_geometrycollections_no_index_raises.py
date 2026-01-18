import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
def test_geometrycollections_no_index_raises():
    with pytest.raises(ValueError):
        shapely.geometrycollections(np.array([point, line_string], dtype=object), indices=[0, 2])