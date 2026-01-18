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
def test_explode_geoseries(self):
    s = GeoSeries([MultiPoint([(0, 0), (1, 1)]), MultiPoint([(2, 2), (3, 3), (4, 4)])], crs=4326)
    s.index.name = 'test_index_name'
    expected_index_name = ['test_index_name', None]
    index = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    expected = GeoSeries([Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)], index=MultiIndex.from_tuples(index, names=expected_index_name), crs=4326)
    with pytest.warns(FutureWarning, match='Currently, index_parts defaults'):
        assert_geoseries_equal(expected, s.explode())