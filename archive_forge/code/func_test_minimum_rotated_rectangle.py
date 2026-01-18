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
def test_minimum_rotated_rectangle(self):
    s = GeoSeries([self.sq, self.t5], crs=3857)
    r = s.minimum_rotated_rectangle()
    exp = GeoSeries.from_wkt(['POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))', 'POLYGON ((2 0, 2 3, 3 3, 3 0, 2 0))'])
    assert np.all(r.normalize().geom_equals_exact(exp, 0.001))
    assert isinstance(r, GeoSeries)
    assert s.crs == r.crs