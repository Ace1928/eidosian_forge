import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_append_before_table_exists(self, engine_postgis, df_nybb):
    """
        Tests that insert works with if_exists='append' when table does not exist yet.
        """
    engine = engine_postgis
    table = 'nybb'
    drop_table_if_exists(engine, table)
    write_postgis(df_nybb, con=engine, name=table, if_exists='append')
    sql = text('SELECT * FROM {table};'.format(table=table))
    df = read_postgis(sql, engine, geom_col='geometry')
    validate_boro_df(df)