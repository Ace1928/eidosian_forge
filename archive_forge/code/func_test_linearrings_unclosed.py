import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_linearrings_unclosed():
    actual = shapely.linearrings(box_tpl(0, 0, 1, 1)[:-1])
    assert_geometries_equal(actual, LinearRing([(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]))