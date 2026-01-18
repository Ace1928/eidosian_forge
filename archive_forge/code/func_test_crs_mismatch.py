import math
from typing import Sequence
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, GeometryCollection
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sjoin, sjoin_nearest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
@pytest.mark.parametrize('dfs', ['default-index', 'string-index'], indirect=True)
def test_crs_mismatch(self, dfs):
    index, df1, df2, expected = dfs
    df1.crs = 'epsg:4326'
    with pytest.warns(UserWarning, match='CRS mismatch between the CRS'):
        sjoin(df1, df2)