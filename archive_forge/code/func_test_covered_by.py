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
@pytest.mark.skipif(not (compat.USE_PYGEOS or compat.USE_SHAPELY_20), reason='covered_by is only implemented for pygeos, not shapely')
def test_covered_by(self):
    res = self.g1.covered_by(self.g1)
    exp = Series([True, True])
    assert_series_equal(res, exp)
    expected = [False, True, True, True, True, True, False, False]
    with pytest.warns(UserWarning, match='The indices .+ different'):
        assert_array_dtype_equal(expected, self.g0.covered_by(self.g9, align=True))
    expected = [False, True, False, False, False, False, False]
    assert_array_dtype_equal(expected, self.g0.covered_by(self.g9, align=False))