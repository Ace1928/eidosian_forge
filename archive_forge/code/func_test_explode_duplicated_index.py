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
def test_explode_duplicated_index(self):
    df = GeoDataFrame({'vals': [1, 2, 3]}, geometry=[MultiPoint([(x, x), (x, 0)]) for x in range(3)], index=[1, 1, 2])
    test_df = df.explode(index_parts=True)
    expected_index = MultiIndex.from_arrays([[1, 1, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
    expected_geometry = GeoSeries([Point(0, 0), Point(0, 0), Point(1, 1), Point(1, 0), Point(2, 2), Point(2, 0)], index=expected_index)
    expected_df = GeoDataFrame({'vals': [1, 1, 2, 2, 3, 3]}, geometry=expected_geometry, index=expected_index)
    assert_geodataframe_equal(test_df, expected_df)