import warnings
import pandas as pd
import pyproj
import pytest
from geopandas._compat import PANDAS_GE_21
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_index_equal
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
def test_concat_axis0_crs(self):
    res = pd.concat([self.gdf, self.gdf])
    self._check_metadata(res)
    res1 = pd.concat([self.gdf.set_crs('epsg:4326'), self.gdf.set_crs('epsg:4326')])
    self._check_metadata(res1, crs='epsg:4326')
    with pytest.warns(UserWarning, match='CRS not set for some of the concatenation inputs.*'):
        res2 = pd.concat([self.gdf, self.gdf.set_crs('epsg:4326')])
        self._check_metadata(res2, crs='epsg:4326')
    with pytest.raises(ValueError, match='Cannot determine common CRS for concatenation inputs.*'):
        pd.concat([self.gdf.set_crs('epsg:4326'), self.gdf.set_crs('epsg:4327')])
    with pytest.warns(UserWarning, match='CRS not set for some of the concatenation inputs.*'):
        res3 = pd.concat([self.gdf, self.gdf.set_crs('epsg:4326'), self.gdf.set_crs('epsg:4326')])
        self._check_metadata(res3, crs='epsg:4326')
    with pytest.raises(ValueError, match='Cannot determine common CRS for concatenation inputs.*'):
        pd.concat([self.gdf, self.gdf.set_crs('epsg:4326'), self.gdf.set_crs('epsg:4327')])