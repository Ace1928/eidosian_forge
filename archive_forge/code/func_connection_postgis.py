import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
@pytest.fixture()
def connection_postgis():
    """
    Initiates a connection to a postGIS database that must already exist.
    See create_postgis for more information.
    """
    psycopg2 = pytest.importorskip('psycopg2')
    from psycopg2 import OperationalError
    dbname = 'test_geopandas'
    user = os.environ.get('PGUSER')
    password = os.environ.get('PGPASSWORD')
    host = os.environ.get('PGHOST')
    port = os.environ.get('PGPORT')
    try:
        con = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    except OperationalError:
        pytest.skip('Cannot connect with postgresql database')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable.*')
        yield con
    con.close()