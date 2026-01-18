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
@pytest.mark.skipif(shapely.geos.geos_version < (3, 11, 0), reason='requires GEOS>=3.11')
@pytest.mark.skipif(not compat.USE_SHAPELY_20, reason='concave_hull is only implemented for shapely >= 2.0')
@pytest.mark.parametrize('expected_series,ratio', [([(0, 0), (0, 3), (1, 1), (3, 3), (3, 0), (0, 0)], 0.0), ([(0, 0), (0, 3), (3, 3), (3, 0), (0, 0)], 1.0)])
def test_concave_hull_accepts_kwargs(self, expected_series, ratio):
    expected = GeoSeries(Polygon(expected_series))
    s = GeoSeries(MultiPoint([(0, 0), (0, 3), (1, 1), (3, 0), (3, 3)]))
    assert_geoseries_equal(expected, s.concave_hull(ratio=ratio))