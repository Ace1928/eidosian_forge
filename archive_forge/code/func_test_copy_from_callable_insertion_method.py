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
@pytest.mark.parametrize('expected_count', [2, 'Success!'])
def test_copy_from_callable_insertion_method(conn, expected_count, request):

    def psql_insert_copy(table, conn, keys, data_iter):
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf = StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)
            columns = ', '.join([f'"{k}"' for k in keys])
            if table.schema:
                table_name = f'{table.schema}.{table.name}'
            else:
                table_name = table.name
            sql_query = f'COPY {table_name} ({columns}) FROM STDIN WITH CSV'
            cur.copy_expert(sql=sql_query, file=s_buf)
        return expected_count
    conn = request.getfixturevalue(conn)
    expected = DataFrame({'col1': [1, 2], 'col2': [0.1, 0.2], 'col3': ['a', 'n']})
    result_count = expected.to_sql(name='test_frame', con=conn, index=False, method=psql_insert_copy)
    if expected_count is None:
        assert result_count is None
    else:
        assert result_count == expected_count
    result = sql.read_sql_table('test_frame', conn)
    tm.assert_frame_equal(result, expected)