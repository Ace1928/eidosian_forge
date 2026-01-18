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
@pytest.mark.parametrize('outer_index', [1, (1, 2), '1'])
def test_explode_pandas_multi_index_ignore_index(self, outer_index):
    index = MultiIndex.from_arrays([[outer_index, outer_index, outer_index], [1, 2, 3]], names=('first', 'second'))
    df = GeoDataFrame({'vals': [1, 2, 3]}, geometry=[MultiPoint([(x, x), (x, 0)]) for x in range(3)], index=index)
    test_df = df.explode(ignore_index=True)
    expected_s = GeoSeries([Point(0, 0), Point(0, 0), Point(1, 1), Point(1, 0), Point(2, 2), Point(2, 0)])
    expected_df = GeoDataFrame({'vals': [1, 1, 2, 2, 3, 3], 'geometry': expected_s})
    expected_index = Index(range(len(expected_df)))
    expected_df = expected_df.set_index(expected_index)
    assert_frame_equal(test_df, expected_df)
    test_df = df.explode(ignore_index=True, index_parts=True)
    assert_frame_equal(test_df, expected_df)