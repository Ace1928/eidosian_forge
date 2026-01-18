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
def test_api_categorical(conn, request):
    if conn == 'postgresql_adbc_conn':
        adbc = import_optional_dependency('adbc_driver_postgresql', errors='ignore')
        if adbc is not None and Version(adbc.__version__) < Version('0.9.0'):
            request.node.add_marker(pytest.mark.xfail(reason='categorical dtype not implemented for ADBC postgres driver', strict=True))
    conn = request.getfixturevalue(conn)
    if sql.has_table('test_categorical', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_categorical')
    df = DataFrame({'person_id': [1, 2, 3], 'person_name': ['John P. Doe', 'Jane Dove', 'John P. Doe']})
    df2 = df.copy()
    df2['person_name'] = df2['person_name'].astype('category')
    df2.to_sql(name='test_categorical', con=conn, index=False)
    res = sql.read_sql_query('SELECT * FROM test_categorical', conn)
    tm.assert_frame_equal(res, df)