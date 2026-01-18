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
@pytest.mark.parametrize('conn', sqlalchemy_connectable_types)
def test_date_parsing(conn, request):
    conn_name = conn
    conn = request.getfixturevalue(conn)
    df = sql.read_sql_table('types', conn)
    expected_type = object if 'sqlite' in conn_name else np.datetime64
    assert issubclass(df.DateCol.dtype.type, expected_type)
    df = sql.read_sql_table('types', conn, parse_dates=['DateCol'])
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    df = sql.read_sql_table('types', conn, parse_dates={'DateCol': '%Y-%m-%d %H:%M:%S'})
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    df = sql.read_sql_table('types', conn, parse_dates={'DateCol': {'format': '%Y-%m-%d %H:%M:%S'}})
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    df = sql.read_sql_table('types', conn, parse_dates=['IntDateCol'])
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    df = sql.read_sql_table('types', conn, parse_dates={'IntDateCol': 's'})
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    df = sql.read_sql_table('types', conn, parse_dates={'IntDateCol': {'unit': 's'}})
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)