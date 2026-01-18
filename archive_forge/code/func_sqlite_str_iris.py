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
def sqlite_str_iris(sqlite_str, iris_path):
    sqlalchemy = pytest.importorskip('sqlalchemy')
    engine = sqlalchemy.create_engine(sqlite_str)
    create_and_load_iris(engine, iris_path)
    create_and_load_iris_view(engine)
    engine.dispose()
    return sqlite_str