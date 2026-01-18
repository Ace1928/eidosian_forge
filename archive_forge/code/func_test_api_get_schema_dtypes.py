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
def test_api_get_schema_dtypes(conn, request):
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'get_schema' not implemented for ADBC drivers", strict=True))
    conn_name = conn
    conn = request.getfixturevalue(conn)
    float_frame = DataFrame({'a': [1.1, 1.2], 'b': [2.1, 2.2]})
    if conn_name == 'sqlite_buildin':
        dtype = 'INTEGER'
    else:
        from sqlalchemy import Integer
        dtype = Integer
    create_sql = sql.get_schema(float_frame, 'test', con=conn, dtype={'b': dtype})
    assert 'CREATE' in create_sql
    assert 'INTEGER' in create_sql