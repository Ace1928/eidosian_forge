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
def test_buffer_distance_array(self):
    original = GeoSeries([self.p0, self.p0])
    expected = GeoSeries([Polygon(((6, 5), (5, 4), (4, 5), (5, 6), (6, 5))), Polygon(((10, 5), (5, 0), (0, 5), (5, 10), (10, 5)))])
    calculated = original.buffer(np.array([1, 5]), resolution=1)
    assert_geoseries_equal(calculated, expected, check_less_precise=True)