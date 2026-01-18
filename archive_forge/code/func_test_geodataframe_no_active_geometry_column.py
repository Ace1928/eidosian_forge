import warnings
import numpy as np
from shapely.geometry import Point, Polygon
import pandas as pd
from pandas import DataFrame, Series
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_geodataframe_no_active_geometry_column():

    def create_dataframe():
        gdf = GeoDataFrame({'value': [1, 2], 'geometry': [Point(1, 1), Point(2, 2)]})
        gdf['geom2'] = GeoSeries([Point(3, 3), Point(4, 4)])
        return gdf
    df1 = create_dataframe()
    df1._geometry_column_name = None
    df2 = create_dataframe()
    df2._geometry_column_name = None
    assert_geodataframe_equal(df1, df2)
    df1 = create_dataframe()[['value', 'geom2']]
    df2 = create_dataframe()[['value', 'geom2']]
    assert_geodataframe_equal(df1, df2)
    df1 = GeoDataFrame(create_dataframe()[['value']])
    df2 = GeoDataFrame(create_dataframe()[['value']])
    assert_geodataframe_equal(df1, df2)