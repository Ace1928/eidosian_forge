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
@pytest.mark.parametrize('conn', all_connectable_types)
@pytest.mark.parametrize('error', ['ignore', 'raise', 'coerce'])
@pytest.mark.parametrize('read_sql, text, mode', [(sql.read_sql, 'SELECT * FROM types', ('sqlalchemy', 'fallback')), (sql.read_sql, 'types', 'sqlalchemy'), (sql.read_sql_query, 'SELECT * FROM types', ('sqlalchemy', 'fallback')), (sql.read_sql_table, 'types', 'sqlalchemy')])
def test_api_custom_dateparsing_error(conn, request, read_sql, text, mode, error, types_data_frame):
    conn_name = conn
    conn = request.getfixturevalue(conn)
    if text == 'types' and conn_name == 'sqlite_buildin_types':
        request.applymarker(pytest.mark.xfail(reason='failing combination of arguments'))
    expected = types_data_frame.astype({'DateCol': 'datetime64[ns]'})
    result = read_sql(text, con=conn, parse_dates={'DateCol': {'errors': error}})
    if 'postgres' in conn_name:
        result['BoolCol'] = result['BoolCol'].astype(int)
        result['BoolColWithNull'] = result['BoolColWithNull'].astype(float)
    if conn_name == 'postgresql_adbc_types':
        expected = expected.astype({'IntDateCol': 'int32', 'IntDateOnlyCol': 'int32', 'IntCol': 'int32'})
        if not pa_version_under13p0:
            expected['DateCol'] = expected['DateCol'].astype('datetime64[us]')
    tm.assert_frame_equal(result, expected)