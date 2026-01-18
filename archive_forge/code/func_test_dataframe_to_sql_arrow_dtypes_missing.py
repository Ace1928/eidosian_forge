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
def test_dataframe_to_sql_arrow_dtypes_missing(conn, request, nulls_fixture):
    pytest.importorskip('pyarrow')
    df = DataFrame({'datetime': pd.array([datetime(2023, 1, 1), nulls_fixture], dtype='timestamp[ns][pyarrow]')})
    conn = request.getfixturevalue(conn)
    df.to_sql(name='test_arrow', con=conn, if_exists='replace', index=False)