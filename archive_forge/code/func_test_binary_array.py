from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
@pytest.mark.parametrize('a', all_types)
@pytest.mark.parametrize('func', BINARY_PREDICATES)
def test_binary_array(a, func):
    with ignore_invalid(shapely.is_empty(a) and shapely.geos_version < (3, 12, 0)):
        actual = func([a, a], point)
    assert actual.shape == (2,)
    assert actual.dtype == np.bool_