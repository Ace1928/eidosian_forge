from __future__ import annotations
import contextlib
from contextlib import closing
import csv
from datetime import (
from io import StringIO
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING
import uuid
import numpy as np
import pytest
from pandas._libs import lib
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.util.version import Version
from pandas.io import sql
from pandas.io.sql import (
@pytest.mark.parametrize('conn', postgresql_connectable)
def test_insertion_method_on_conflict_do_nothing(conn, request):
    conn = request.getfixturevalue(conn)
    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    def insert_on_conflict(table, conn, keys, data_iter):
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(table.table).values(data).on_conflict_do_nothing(index_elements=['a'])
        result = conn.execute(stmt)
        return result.rowcount
    create_sql = text('\n    CREATE TABLE test_insert_conflict (\n        a  integer PRIMARY KEY,\n        b  numeric,\n        c  text\n    );\n    ')
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                con.execute(create_sql)
    else:
        with conn.begin():
            conn.execute(create_sql)
    expected = DataFrame([[1, 2.1, 'a']], columns=list('abc'))
    expected.to_sql(name='test_insert_conflict', con=conn, if_exists='append', index=False)
    df_insert = DataFrame([[1, 3.2, 'b']], columns=list('abc'))
    inserted = df_insert.to_sql(name='test_insert_conflict', con=conn, index=False, if_exists='append', method=insert_on_conflict)
    result = sql.read_sql_table('test_insert_conflict', conn)
    tm.assert_frame_equal(result, expected)
    assert inserted == 0
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table('test_insert_conflict')