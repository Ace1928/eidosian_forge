from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
@pytest.mark.parametrize('func', BINARY_PREDICATES)
def test_binary_missing(func):
    actual = func(np.array([point, None, None]), np.array([None, point, None]))
    assert (~actual).all()