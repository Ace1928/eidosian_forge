import string
import warnings
import numpy as np
from numpy.testing import assert_array_equal
from pandas import DataFrame, Index, MultiIndex, Series, concat
import shapely
from shapely.geometry import (
from shapely.geometry.collection import GeometryCollection
from shapely.ops import unary_union
from shapely import wkt
from geopandas import GeoDataFrame, GeoSeries
from geopandas.base import GeoPandasBase
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
from geopandas.tests.util import assert_geoseries_equal, geom_equals
from geopandas import _compat as compat
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest
def test_geo_op_empty_result(self):
    l1 = LineString([(0, 0), (1, 1)])
    l2 = LineString([(2, 2), (3, 3)])
    expected = GeoSeries([GeometryCollection()])
    result = GeoSeries([l1]).intersection(l2)
    assert_geoseries_equal(result, expected)
    result = GeoSeries([l1]).intersection(GeoSeries([l2]))
    assert_geoseries_equal(result, expected)
    result = GeoSeries([GeometryCollection()]).convex_hull
    assert_geoseries_equal(result, expected)