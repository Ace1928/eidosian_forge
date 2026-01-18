import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
@pytest.fixture(params=['default-index', 'int-index', 'string-index'])
def dfs_index(request, dfs):
    df1, df2 = dfs
    if request.param == 'int-index':
        df1.index = [1, 2]
        df2.index = [0, 2]
    if request.param == 'string-index':
        df1.index = ['row1', 'row2']
    return (df1, df2)