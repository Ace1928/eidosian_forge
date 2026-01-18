import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
def test_hausdorff_distance_densify():
    a = shapely.linestrings([[0, 0], [100, 0], [10, 100], [10, 100]])
    b = shapely.linestrings([[0, 100], [0, 10], [80, 10]])
    with ignore_invalid(shapely.geos_version < (3, 12, 0)):
        actual = shapely.hausdorff_distance(a, b, densify=0.001)
    assert actual == pytest.approx(47.8, abs=0.1)