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
@pytest.mark.skipif(not (compat.USE_PYGEOS and compat.SHAPELY_GE_20), reason='concave_hull is only implemented for shapely >= 2.0')
def test_concave_hull_pygeos_set_shapely_installed(self):
    expected = GeoSeries([Polygon([(0, 1), (1, 1), (0, 0), (0, 1)]), Polygon([(1, 0), (0, 0), (0, 1), (1, 1), (1, 0)])])
    with pytest.warns(UserWarning, match='PyGEOS does not support concave_hull, and Shapely >= 2 is installed'):
        assert_geoseries_equal(expected, self.g5.concave_hull())