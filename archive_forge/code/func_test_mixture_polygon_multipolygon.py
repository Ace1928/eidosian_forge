import numpy as np
import pytest
from numpy.testing import assert_allclose
import shapely
from shapely import MultiLineString, MultiPoint, MultiPolygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_mixture_polygon_multipolygon():
    typ, coords, offsets = shapely.to_ragged_array([polygon, multi_polygon])
    assert typ == shapely.GeometryType.MULTIPOLYGON
    result = shapely.from_ragged_array(typ, coords, offsets)
    expected = np.array([MultiPolygon([polygon]), multi_polygon])
    assert_geometries_equal(result, expected)