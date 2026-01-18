import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_append_with_different_crs(self, engine_postgis, df_nybb):
    """
        Tests that the warning is raised if table CRS differs from frame.
        """
    engine = engine_postgis
    table = 'nybb'
    write_postgis(df_nybb, con=engine, name=table, if_exists='replace')
    df_nybb2 = df_nybb.to_crs(epsg=4326)
    with pytest.raises(ValueError, match='CRS of the target table'):
        write_postgis(df_nybb2, con=engine, name=table, if_exists='append')