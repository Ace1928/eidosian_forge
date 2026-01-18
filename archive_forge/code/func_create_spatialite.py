import os.path
from pandas import Series
from geopandas import GeoDataFrame
from geopandas.testing import (  # noqa: F401
def create_spatialite(con, df):
    """
    Return a SpatiaLite connection containing the nybb table.

    Parameters
    ----------
    `con`: ``sqlite3.Connection``
    `df`: ``GeoDataFrame``
    """
    with con:
        geom_col = df.geometry.name
        srid = get_srid(df)
        con.execute('CREATE TABLE IF NOT EXISTS nybb ( ogc_fid INTEGER PRIMARY KEY, borocode INTEGER, boroname TEXT, shape_leng REAL, shape_area REAL)')
        con.execute('SELECT AddGeometryColumn(?, ?, ?, ?)', ('nybb', geom_col, srid, df.geom_type.dropna().iat[0].upper()))
        con.execute('SELECT CreateSpatialIndex(?, ?)', ('nybb', geom_col))
        sql_row = 'INSERT INTO nybb VALUES(?, ?, ?, ?, ?, GeomFromText(?, ?))'
        con.executemany(sql_row, ((None, row.BoroCode, row.BoroName, row.Shape_Leng, row.Shape_Area, row.geometry.wkt if row.geometry else None, srid) for row in df.itertuples(index=False)))