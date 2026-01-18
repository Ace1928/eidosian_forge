import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_from_postgis_custom_geom_col(self, connection_postgis, df_nybb):
    con = connection_postgis
    geom_col = 'the_geom'
    create_postgis(con, df_nybb, geom_col=geom_col)
    sql = 'SELECT * FROM nybb;'
    df = GeoDataFrame.from_postgis(sql, con, geom_col=geom_col)
    validate_boro_df(df, case_sensitive=False)