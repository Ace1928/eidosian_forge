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
@pytest.mark.skipif(not (compat.USE_PYGEOS or compat.USE_SHAPELY_20), reason='delaunay_triangles not implemented for shapely<2')
def test_delaunay_triangles_pass_kwargs(self):
    expected = GeoSeries([MultiLineString([[(0, 0), (1, 1)], [(0, 0), (1, 0)], [(1, 0), (1, 1)]]), MultiLineString([[(0, 1), (1, 1)], [(0, 0), (0, 1)], [(0, 0), (1, 1)]])])
    dlt = self.g3.delaunay_triangles(only_edges=True)
    assert isinstance(dlt, GeoSeries)
    assert_series_equal(expected, dlt)