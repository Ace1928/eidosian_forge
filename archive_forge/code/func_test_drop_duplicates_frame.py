import os
from packaging.version import Version
import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import shapely
from shapely.geometry import Point, GeometryCollection, LineString, LinearRing
import geopandas
from geopandas import GeoDataFrame, GeoSeries
import geopandas._compat as compat
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
@pytest.mark.xfail(strict=False)
def test_drop_duplicates_frame():
    gdf_len = 3
    dup_gdf = GeoDataFrame({'geometry': [Point(0, 0) for _ in range(gdf_len)], 'value1': range(gdf_len)})
    dropped_geometry = dup_gdf.drop_duplicates(subset='geometry')
    assert len(dropped_geometry) == 1
    dropped_all = dup_gdf.drop_duplicates()
    assert len(dropped_all) == gdf_len