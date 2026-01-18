import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
@pytest.mark.parametrize('op, kwargs', [pytest.param('dwithin', dict(distance=0.5), marks=pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='GEOS < 3.10')), ('equals_exact', dict(tolerance=0.01)), ('relate_pattern', dict(pattern='T*F**F***'))])
def test_array_argument_binary_predicates2(op, kwargs):
    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    points = shapely.points([(0, 0), (0.5, 0.5), (1, 1)])
    result = getattr(polygon, op)(points, **kwargs)
    assert isinstance(result, np.ndarray)
    expected = np.array([getattr(polygon, op)(p, **kwargs) for p in points], dtype=bool)
    np.testing.assert_array_equal(result, expected)