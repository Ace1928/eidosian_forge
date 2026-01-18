import warnings
import pandas as pd
import pyproj
import pytest
from geopandas._compat import PANDAS_GE_21
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_index_equal
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
def test_concat_axis0_unaligned_cols(self):
    gdf = self.gdf.set_crs('epsg:4326').assign(geom=self.gdf.geometry.set_crs('epsg:4327'))
    both_geom_cols = gdf[['geom', 'geometry']]
    single_geom_col = gdf[['geometry']]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        pd.concat([both_geom_cols, single_geom_col])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        pd.concat([single_geom_col, both_geom_cols])
    explicit_all_none_case = gdf[['geometry']].assign(geom=GeoSeries([None for _ in range(len(gdf))]))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        pd.concat([both_geom_cols, explicit_all_none_case])
    with pytest.warns(UserWarning, match='CRS not set for some of the concatenation inputs.*'):
        partial_none_case = self.gdf[['geometry']]
        partial_none_case.iloc[0] = None
        pd.concat([single_geom_col, partial_none_case])