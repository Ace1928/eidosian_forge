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
@pytest.mark.parametrize('conn', all_connectable)
def test_api_roundtrip(conn, request, test_frame1):
    conn_name = conn
    conn = request.getfixturevalue(conn)
    if sql.has_table('test_frame_roundtrip', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_frame_roundtrip')
    sql.to_sql(test_frame1, 'test_frame_roundtrip', con=conn)
    result = sql.read_sql_query('SELECT * FROM test_frame_roundtrip', con=conn)
    if 'adbc' in conn_name:
        result = result.rename(columns={'__index_level_0__': 'level_0'})
    result.index = test_frame1.index
    result.set_index('level_0', inplace=True)
    result.index.astype(int)
    result.index.name = None
    tm.assert_frame_equal(result, test_frame1)