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
def mysql_pymysql_engine():
    sqlalchemy = pytest.importorskip('sqlalchemy')
    pymysql = pytest.importorskip('pymysql')
    engine = sqlalchemy.create_engine('mysql+pymysql://root@localhost:3306/pandas', connect_args={'client_flag': pymysql.constants.CLIENT.MULTI_STATEMENTS}, poolclass=sqlalchemy.pool.NullPool)
    yield engine
    for view in get_all_views(engine):
        drop_view(view, engine)
    for tbl in get_all_tables(engine):
        drop_table(tbl, engine)
    engine.dispose()