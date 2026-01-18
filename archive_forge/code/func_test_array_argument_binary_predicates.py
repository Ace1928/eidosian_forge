import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
@pytest.mark.parametrize('op', ['crosses', 'contains', 'contains_properly', 'covered_by', 'covers', 'disjoint', 'equals', 'intersects', 'overlaps', 'touches', 'within'])
def test_array_argument_binary_predicates(op):
    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    points = shapely.points([(0, 0), (0.5, 0.5), (1, 1)])
    result = getattr(polygon, op)(points)
    assert isinstance(result, np.ndarray)
    expected = np.array([getattr(polygon, op)(p) for p in points], dtype=bool)
    np.testing.assert_array_equal(result, expected)