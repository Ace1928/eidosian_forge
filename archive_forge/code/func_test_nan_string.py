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
def test_nan_string(conn, request):
    conn = request.getfixturevalue(conn)
    df = DataFrame({'A': [0, 1, 2], 'B': ['a', 'b', np.nan]})
    assert df.to_sql(name='test_nan', con=conn, index=False) == 3
    df.loc[2, 'B'] = None
    result = sql.read_sql_table('test_nan', conn)
    tm.assert_frame_equal(result, df)
    result = sql.read_sql_query('SELECT * FROM test_nan', conn)
    tm.assert_frame_equal(result, df)