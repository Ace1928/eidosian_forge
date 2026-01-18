import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
def test_distance_missing():
    actual = shapely.distance(point, None)
    assert np.isnan(actual)