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
@pytest.fixture
def postgresql_adbc_conn():
    pytest.importorskip('adbc_driver_postgresql')
    from adbc_driver_postgresql import dbapi
    uri = 'postgresql://postgres:postgres@localhost:5432/pandas'
    with dbapi.connect(uri) as conn:
        yield conn
        for view in get_all_views(conn):
            drop_view(view, conn)
        for tbl in get_all_tables(conn):
            drop_table(tbl, conn)
        conn.commit()