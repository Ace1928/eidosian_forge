import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_linestrings_from_xy_broadcast():
    x = [0, 1]
    y = ([2, 3], [4, 5])
    actual = shapely.linestrings(x, y)
    assert_geometries_equal(actual, [LineString([(0, 2), (1, 3)]), LineString([(0, 4), (1, 5)])])