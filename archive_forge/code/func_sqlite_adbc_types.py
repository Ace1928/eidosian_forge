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
def sqlite_adbc_types(sqlite_adbc_conn, types_data):
    import adbc_driver_manager as mgr
    conn = sqlite_adbc_conn
    try:
        conn.adbc_get_table_schema('types')
    except mgr.ProgrammingError:
        conn.rollback()
        new_data = []
        for entry in types_data:
            entry['BoolCol'] = int(entry['BoolCol'])
            if entry['BoolColWithNull'] is not None:
                entry['BoolColWithNull'] = int(entry['BoolColWithNull'])
            new_data.append(tuple(entry.values()))
        create_and_load_types_sqlite3(conn, new_data)
        conn.commit()
    yield conn