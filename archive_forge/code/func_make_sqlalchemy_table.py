import datetime
import logging
from petl.compat import long, text_type
from petl.errors import ArgumentError
from petl.util.materialise import columns
from petl.transform.basics import head
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
def make_sqlalchemy_table(table, tablename, schema=None, constraints=True, metadata=None):
    """
    Create an SQLAlchemy table definition based on data in `table`.

    Keyword arguments:

    table : table container
        Table data to use to infer types etc.
    tablename : text
        Name of the table
    schema : text
        Name of the database schema to create the table in
    constraints : bool
        If True use length and nullable constraints
    metadata : sqlalchemy.MetaData
        Custom table metadata

    """
    import sqlalchemy
    if not metadata:
        metadata = sqlalchemy.MetaData()
    sql_table = sqlalchemy.Table(tablename, metadata, schema=schema)
    cols = columns(table)
    flds = list(cols.keys())
    for f in flds:
        sql_column = make_sqlalchemy_column(cols[f], f, constraints=constraints)
        sql_table.append_column(sql_column)
    return sql_table