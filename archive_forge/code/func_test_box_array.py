import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('coords,ccw,expected', [((0, 0, [1, 2], [1, 2]), True, [Polygon([(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]), Polygon([(2, 0), (2, 2), (0, 2), (0, 0), (2, 0)])]), ((0, 0, [1, 2], [1, 2]), [True, False], [Polygon([(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]), Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)])])])
def test_box_array(coords, ccw, expected):
    actual = shapely.box(*coords, ccw=ccw)
    assert_geometries_equal(actual, expected)