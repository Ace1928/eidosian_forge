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
def test_api_to_sql_append(conn, request, test_frame1):
    conn = request.getfixturevalue(conn)
    if sql.has_table('test_frame4', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_frame4')
    assert sql.to_sql(test_frame1, 'test_frame4', conn, if_exists='fail') == 4
    assert sql.to_sql(test_frame1, 'test_frame4', conn, if_exists='append') == 4
    assert sql.has_table('test_frame4', conn)
    num_entries = 2 * len(test_frame1)
    num_rows = count_rows(conn, 'test_frame4')
    assert num_rows == num_entries