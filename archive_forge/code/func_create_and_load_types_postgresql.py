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
def create_and_load_types_postgresql(conn, types_data: list[dict]):
    with conn.cursor() as cur:
        stmt = 'CREATE TABLE types (\n                        "TextCol" TEXT,\n                        "DateCol" TIMESTAMP,\n                        "IntDateCol" INTEGER,\n                        "IntDateOnlyCol" INTEGER,\n                        "FloatCol" DOUBLE PRECISION,\n                        "IntCol" INTEGER,\n                        "BoolCol" BOOLEAN,\n                        "IntColWithNull" INTEGER,\n                        "BoolColWithNull" BOOLEAN\n                    )'
        cur.execute(stmt)
        stmt = '\n                INSERT INTO types\n                VALUES($1, $2::timestamp, $3, $4, $5, $6, $7, $8, $9)\n                '
        cur.executemany(stmt, types_data)
    conn.commit()