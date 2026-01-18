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
def create_and_load_iris_postgresql(conn, iris_file: Path):
    stmt = 'CREATE TABLE iris (\n            "SepalLength" DOUBLE PRECISION,\n            "SepalWidth" DOUBLE PRECISION,\n            "PetalLength" DOUBLE PRECISION,\n            "PetalWidth" DOUBLE PRECISION,\n            "Name" TEXT\n        )'
    with conn.cursor() as cur:
        cur.execute(stmt)
        with iris_file.open(newline=None, encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            stmt = 'INSERT INTO iris VALUES($1, $2, $3, $4, $5)'
            records = [(float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4]) for row in reader]
            cur.executemany(stmt, records)
    conn.commit()