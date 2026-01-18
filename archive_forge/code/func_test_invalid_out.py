import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('func', [shapely.points, shapely.linestrings, shapely.geometrycollections])
@pytest.mark.parametrize('out', [[None, None, None], np.empty(3), non_writeable, np.empty((3, 2), dtype=object), np.empty((), dtype=object), np.empty((2,), dtype=object)])
def test_invalid_out(func, out):
    if func is shapely.points:
        x = [[0.2, 0.3], [0.4, 0.5]]
        indices = [0, 2]
    elif func is shapely.linestrings:
        x = [[1, 1], [2, 1], [2, 2], [3, 3], [3, 4], [4, 4]]
        indices = [0, 0, 0, 2, 2, 2]
    else:
        x = [point, line_string]
        indices = [0, 2]
    with pytest.raises((TypeError, ValueError)):
        func(x, indices=indices, out=out)