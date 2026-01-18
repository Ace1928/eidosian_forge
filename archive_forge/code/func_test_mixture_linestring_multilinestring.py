import numpy as np
import pytest
from numpy.testing import assert_allclose
import shapely
from shapely import MultiLineString, MultiPoint, MultiPolygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_mixture_linestring_multilinestring():
    typ, coords, offsets = shapely.to_ragged_array([line_string, multi_line_string])
    assert typ == shapely.GeometryType.MULTILINESTRING
    result = shapely.from_ragged_array(typ, coords, offsets)
    expected = np.array([MultiLineString([line_string]), multi_line_string])
    assert_geometries_equal(result, expected)