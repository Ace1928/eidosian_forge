import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('func,expected', [(shapely.multipoints, MultiPoint()), (shapely.multilinestrings, MultiLineString()), (shapely.multipolygons, MultiPolygon()), (shapely.geometrycollections, GeometryCollection())])
def test_create_collection_only_none(func, expected):
    actual = func(np.array([None], dtype=object))
    assert_geometries_equal(actual, expected)