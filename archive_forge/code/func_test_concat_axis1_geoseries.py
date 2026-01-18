import warnings
import pandas as pd
import pyproj
import pytest
from geopandas._compat import PANDAS_GE_21
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_index_equal
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
@pytest.mark.filterwarnings('ignore:Accessing CRS')
def test_concat_axis1_geoseries(self):
    gseries2 = GeoSeries([Point(i, i) for i in range(3, 6)], crs='epsg:4326')
    result = pd.concat([gseries2, self.gseries], axis=1)
    assert type(result) is GeoDataFrame
    assert result._geometry_column_name is None
    assert_index_equal(pd.Index([0, 1]), result.columns)
    gseries2.name = 'foo'
    result2 = pd.concat([gseries2, self.gseries], axis=1)
    assert type(result2) is GeoDataFrame
    assert result._geometry_column_name is None
    assert_index_equal(pd.Index(['foo', 0]), result2.columns)