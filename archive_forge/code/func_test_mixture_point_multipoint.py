import numpy as np
import pytest
from numpy.testing import assert_allclose
import shapely
from shapely import MultiLineString, MultiPoint, MultiPolygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_mixture_point_multipoint():
    typ, coords, offsets = shapely.to_ragged_array([point, multi_point])
    assert typ == shapely.GeometryType.MULTIPOINT
    result = shapely.from_ragged_array(typ, coords, offsets)
    expected = np.array([MultiPoint([point]), multi_point])
    assert_geometries_equal(result, expected)