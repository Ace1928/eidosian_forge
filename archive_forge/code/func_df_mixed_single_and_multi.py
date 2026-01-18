import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
@pytest.fixture
def df_mixed_single_and_multi():
    from shapely.geometry import Point, LineString, MultiLineString
    df = geopandas.GeoDataFrame({'geometry': [LineString([(0, 0), (1, 1)]), MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]), Point(0, 1)]}, crs='epsg:4326')
    return df