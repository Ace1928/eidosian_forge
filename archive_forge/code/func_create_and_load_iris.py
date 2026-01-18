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
def create_and_load_iris(conn, iris_file: Path):
    from sqlalchemy import insert
    iris = iris_table_metadata()
    with iris_file.open(newline=None, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        params = [dict(zip(header, row)) for row in reader]
        stmt = insert(iris).values(params)
        with conn.begin() as con:
            iris.drop(con, checkfirst=True)
            iris.create(bind=con)
            con.execute(stmt)