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
def test_api_to_sql_index_label_multiindex(conn, request):
    conn_name = conn
    if 'mysql' in conn_name:
        request.applymarker(pytest.mark.xfail(reason='MySQL can fail using TEXT without length as key', strict=False))
    elif 'adbc' in conn_name:
        request.node.add_marker(pytest.mark.xfail(reason='index_label argument NotImplemented with ADBC'))
    conn = request.getfixturevalue(conn)
    if sql.has_table('test_index_label', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_index_label')
    expected_row_count = 4
    temp_frame = DataFrame({'col1': range(4)}, index=MultiIndex.from_product([('A0', 'A1'), ('B0', 'B1')]))
    result = sql.to_sql(temp_frame, 'test_index_label', conn)
    assert result == expected_row_count
    frame = sql.read_sql_query('SELECT * FROM test_index_label', conn)
    assert frame.columns[0] == 'level_0'
    assert frame.columns[1] == 'level_1'
    result = sql.to_sql(temp_frame, 'test_index_label', conn, if_exists='replace', index_label=['A', 'B'])
    assert result == expected_row_count
    frame = sql.read_sql_query('SELECT * FROM test_index_label', conn)
    assert frame.columns[:2].tolist() == ['A', 'B']
    temp_frame.index.names = ['A', 'B']
    result = sql.to_sql(temp_frame, 'test_index_label', conn, if_exists='replace')
    assert result == expected_row_count
    frame = sql.read_sql_query('SELECT * FROM test_index_label', conn)
    assert frame.columns[:2].tolist() == ['A', 'B']
    result = sql.to_sql(temp_frame, 'test_index_label', conn, if_exists='replace', index_label=['C', 'D'])
    assert result == expected_row_count
    frame = sql.read_sql_query('SELECT * FROM test_index_label', conn)
    assert frame.columns[:2].tolist() == ['C', 'D']
    msg = "Length of 'index_label' should match number of levels, which is 2"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(temp_frame, 'test_index_label', conn, if_exists='replace', index_label='C')