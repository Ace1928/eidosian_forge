import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import Point, Polygon
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, points_from_xy, read_file
from geopandas.array import GeometryArray, GeometryDtype, from_shapely
from geopandas._compat import ignore_shapely2_warnings
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest
@pytest.mark.parametrize('crs', ['+proj=cea +lon_0=0 +lat_ts=45 +x_0=0 +y_0=0 +ellps=WGS84 +units=m', 'IGNF:WGS84'])
def test_geodataframe_crs_nonrepresentable_json(crs):
    gdf = GeoDataFrame([Point(1000, 1000)], columns=['geometry'], crs=crs)
    with pytest.warns(UserWarning, match="GeoDataFrame's CRS is not representable in URN OGC"):
        gdf_geojson = json.loads(gdf.to_json())
    assert 'crs' not in gdf_geojson