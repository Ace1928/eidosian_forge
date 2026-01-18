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
@pytest.mark.skipif(compat.USE_PYGEOS or compat.USE_SHAPELY_20, reason='extract_unique_points not implemented for shapely<2')
def test_extract_unique_points_not_implemented(self):
    with pytest.raises(NotImplementedError, match='shapely >= 2.0 or PyGEOS is required'):
        self.g1.extract_unique_points()