import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
def test_array_argument_buffer():
    point = Point(1, 1)
    distances = np.array([0, 0.5, 1])
    result = point.buffer(distances)
    assert isinstance(result, np.ndarray)
    expected = np.array([point.buffer(d) for d in distances], dtype=object)
    assert_geometries_equal(result, expected)