import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_linestrings_from_coords():
    actual = shapely.linestrings([[[0, 0], [1, 1]], [[0, 0], [2, 2]]])
    assert_geometries_equal(actual, [LineString([(0, 0), (1, 1)]), LineString([(0, 0), (2, 2)])])