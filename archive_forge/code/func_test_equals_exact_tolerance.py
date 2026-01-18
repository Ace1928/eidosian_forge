from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
def test_equals_exact_tolerance():
    p1 = shapely.points(50, 4)
    p2 = shapely.points(50.1, 4.1)
    actual = shapely.equals_exact([p1, p2, None], p1, tolerance=0.05)
    np.testing.assert_allclose(actual, [True, False, False])
    assert actual.dtype == np.bool_
    actual = shapely.equals_exact([p1, p2, None], p1, tolerance=0.2)
    np.testing.assert_allclose(actual, [True, True, False])
    assert actual.dtype == np.bool_
    assert shapely.equals_exact(p1, p1).item() is True
    assert shapely.equals_exact(p1, p2).item() is False
    actual = shapely.equals_exact(p1, p2, tolerance=[0.05, 0.2, np.nan])
    np.testing.assert_allclose(actual, [False, True, False])