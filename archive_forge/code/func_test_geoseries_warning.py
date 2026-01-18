import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_geoseries_warning(dfs):
    df1, df2 = dfs
    with pytest.raises(NotImplementedError):
        overlay(df1, df2.geometry, how='union')