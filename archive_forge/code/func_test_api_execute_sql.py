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
@pytest.mark.parametrize('conn', all_connectable_iris)
def test_api_execute_sql(conn, request):
    conn = request.getfixturevalue(conn)
    with sql.pandasSQL_builder(conn) as pandas_sql:
        iris_results = pandas_sql.execute('SELECT * FROM iris')
        row = iris_results.fetchone()
        iris_results.close()
    assert list(row) == [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']