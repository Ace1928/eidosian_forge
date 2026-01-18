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
def test_api_to_sql_series(conn, request):
    conn = request.getfixturevalue(conn)
    if sql.has_table('test_series', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_series')
    s = Series(np.arange(5, dtype='int64'), name='series')
    sql.to_sql(s, 'test_series', conn, index=False)
    s2 = sql.read_sql_query('SELECT * FROM test_series', conn)
    tm.assert_frame_equal(s.to_frame(), s2)