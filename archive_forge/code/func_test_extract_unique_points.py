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
@pytest.mark.skipif(not (compat.USE_PYGEOS or compat.USE_SHAPELY_20), reason='extract_unique_points is only implemented for pygeos and shapely >= 2.0')
def test_extract_unique_points(self):
    eup = GeoSeries([self.t6]).extract_unique_points()
    expected = GeoSeries([MultiPoint([(2, 0), (3, 0)])])
    assert_series_equal(eup, expected)