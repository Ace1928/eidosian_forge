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
@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_get_schema_create_table(conn, request, test_frame3):
    if conn == 'sqlite_str':
        request.applymarker(pytest.mark.xfail(reason='test does not support sqlite_str fixture'))
    conn = request.getfixturevalue(conn)
    from sqlalchemy import text
    from sqlalchemy.engine import Engine
    tbl = 'test_get_schema_create_table'
    create_sql = sql.get_schema(test_frame3, tbl, con=conn)
    blank_test_df = test_frame3.iloc[:0]
    create_sql = text(create_sql)
    if isinstance(conn, Engine):
        with conn.connect() as newcon:
            with newcon.begin():
                newcon.execute(create_sql)
    else:
        conn.execute(create_sql)
    returned_df = sql.read_sql_table(tbl, conn)
    tm.assert_frame_equal(returned_df, blank_test_df, check_index_type=False)