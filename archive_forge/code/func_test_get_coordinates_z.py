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
@pytest.mark.skipif(not (compat.USE_PYGEOS or compat.USE_SHAPELY_20), reason='get_coordinates not implemented for shapely<2')
def test_get_coordinates_z(self):
    expected = DataFrame(data=self.expected_3d, columns=['x', 'y', 'z'], index=[0, 1, 3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6])
    assert_frame_equal(self.g11.get_coordinates(include_z=True), expected)