import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_linestrings_from_xy():
    actual = shapely.linestrings([0, 1], [2, 3])
    assert_geometries_equal(actual, LineString([(0, 2), (1, 3)]))