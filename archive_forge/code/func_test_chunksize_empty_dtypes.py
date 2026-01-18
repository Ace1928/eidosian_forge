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
def test_chunksize_empty_dtypes(conn, request):
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason='chunksize argument NotImplemented with ADBC'))
    conn = request.getfixturevalue(conn)
    dtypes = {'a': 'int64', 'b': 'object'}
    df = DataFrame(columns=['a', 'b']).astype(dtypes)
    expected = df.copy()
    df.to_sql(name='test', con=conn, index=False, if_exists='replace')
    for result in read_sql_query('SELECT * FROM test', conn, dtype=dtypes, chunksize=1):
        tm.assert_frame_equal(result, expected)