import warnings
import pandas as pd
import pyproj
import pytest
from geopandas._compat import PANDAS_GE_21
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_index_equal
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
def test_concat_axis1_multiple_geodataframes(self):
    if PANDAS_GE_21:
        expected_err = "Concat operation has resulted in multiple columns using the geometry column name 'geometry'."
    else:
        expected_err = "GeoDataFrame does not support multiple columns using the geometry column name 'geometry'"
    with pytest.raises(ValueError, match=expected_err):
        pd.concat([self.gdf, self.gdf], axis=1)
    df2 = self.gdf.rename_geometry('geom')
    expected_err2 = "Concat operation has resulted in multiple columns using the geometry column name 'geom'."
    with pytest.raises(ValueError, match=expected_err2):
        pd.concat([df2, df2], axis=1)
    res3 = pd.concat([df2.set_crs('epsg:4326'), self.gdf], axis=1)
    self._check_metadata(res3, geometry_column_name='geom', crs='epsg:4326')