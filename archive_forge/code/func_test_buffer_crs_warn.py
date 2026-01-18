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
def test_buffer_crs_warn(self):
    with pytest.warns(UserWarning, match='Geometry is in a geographic CRS'):
        self.g4.buffer(1)
    with warnings.catch_warnings(record=True) as record:
        self.g4.buffer(0)
    for r in record:
        assert 'Geometry is in a geographic CRS.' not in str(r.message)