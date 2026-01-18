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
def create_and_load_postgres_datetz(conn):
    from sqlalchemy import Column, DateTime, MetaData, Table, insert
    from sqlalchemy.engine import Engine
    metadata = MetaData()
    datetz = Table('datetz', metadata, Column('DateColWithTz', DateTime(timezone=True)))
    datetz_data = [{'DateColWithTz': '2000-01-01 00:00:00-08:00'}, {'DateColWithTz': '2000-06-01 00:00:00-07:00'}]
    stmt = insert(datetz).values(datetz_data)
    if isinstance(conn, Engine):
        with conn.connect() as conn:
            with conn.begin():
                datetz.drop(conn, checkfirst=True)
                datetz.create(bind=conn)
                conn.execute(stmt)
    else:
        with conn.begin():
            datetz.drop(conn, checkfirst=True)
            datetz.create(bind=conn)
            conn.execute(stmt)
    expected_data = [Timestamp('2000-01-01 08:00:00', tz='UTC'), Timestamp('2000-06-01 07:00:00', tz='UTC')]
    return Series(expected_data, name='DateColWithTz')