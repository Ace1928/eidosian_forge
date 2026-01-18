from __future__ import absolute_import, print_function, division
import logging
from petl.compat import next, text_type, string_types
from petl.errors import ArgumentError
from petl.util.base import Table
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
from petl.io.db_create import drop_table, create_table
def appenddb(table, dbo, tablename, schema=None, commit=True):
    """
    Load data into an existing database table via a DB-API 2.0
    connection or cursor. As :func:`petl.io.db.todb` except that the database
    table will be appended, i.e., the new data will be inserted into the
    table, and any existing rows will remain.

    """
    needs_closing = False
    if isinstance(dbo, string_types):
        import sqlite3
        dbo = sqlite3.connect(dbo)
        needs_closing = True
    try:
        _todb(table, dbo, tablename, schema=schema, commit=commit, truncate=False)
    finally:
        if needs_closing:
            dbo.close()