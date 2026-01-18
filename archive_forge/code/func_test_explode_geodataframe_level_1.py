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
@pytest.mark.parametrize('index_name', [None, 'test'])
def test_explode_geodataframe_level_1(self, index_name):
    s = GeoSeries([MultiPoint([Point(1, 2), Point(2, 3)]), Point(5, 5)])
    df = GeoDataFrame({'level_1': [1, 2], 'geometry': s})
    df.index.name = index_name
    test_df = df.explode(index_parts=True)
    expected_s = GeoSeries([Point(1, 2), Point(2, 3), Point(5, 5)])
    expected_df = GeoDataFrame({'level_1': [1, 1, 2], 'geometry': expected_s})
    expected_index = MultiIndex([[0, 1], [0, 1]], [[0, 0, 1], [0, 1, 0]], names=[index_name, None])
    expected_df = expected_df.set_index(expected_index)
    assert_frame_equal(test_df, expected_df)