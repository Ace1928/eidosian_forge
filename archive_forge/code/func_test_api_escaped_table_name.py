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
def test_api_escaped_table_name(conn, request):
    conn_name = conn
    conn = request.getfixturevalue(conn)
    if sql.has_table('d1187b08-4943-4c8d-a7f6', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('d1187b08-4943-4c8d-a7f6')
    df = DataFrame({'A': [0, 1, 2], 'B': [0.2, np.nan, 5.6]})
    df.to_sql(name='d1187b08-4943-4c8d-a7f6', con=conn, index=False)
    if 'postgres' in conn_name:
        query = 'SELECT * FROM "d1187b08-4943-4c8d-a7f6"'
    else:
        query = 'SELECT * FROM `d1187b08-4943-4c8d-a7f6`'
    res = sql.read_sql_query(query, conn)
    tm.assert_frame_equal(res, df)