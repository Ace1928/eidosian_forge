import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.tests.common import all_types, all_types_z, ignore_invalid
@pytest.mark.parametrize('left, right', cases1)
def test_equality_with_nan(left, right):
    assert not left == right
    assert left != right