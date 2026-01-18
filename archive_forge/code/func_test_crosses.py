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
def test_crosses(self):
    expected = [False, False, False, False, False, False, False]
    assert_array_dtype_equal(expected, self.g0.crosses(self.t1))
    expected = [False, True]
    assert_array_dtype_equal(expected, self.crossed_lines.crosses(self.l3))
    expected = [False] * 8
    with pytest.warns(UserWarning, match='The indices .+ different'):
        assert_array_dtype_equal(expected, self.g0.crosses(self.g9, align=True))
    expected = [False] * 7
    assert_array_dtype_equal(expected, self.g0.crosses(self.g9, align=False))