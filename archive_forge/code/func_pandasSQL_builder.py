from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
def pandasSQL_builder(con, schema: str | None=None, need_transaction: bool=False) -> PandasSQL:
    """
    Convenience function to return the correct PandasSQL subclass based on the
    provided parameters.  Also creates a sqlalchemy connection and transaction
    if necessary.
    """
    import sqlite3
    if isinstance(con, sqlite3.Connection) or con is None:
        return SQLiteDatabase(con)
    sqlalchemy = import_optional_dependency('sqlalchemy', errors='ignore')
    if isinstance(con, str) and sqlalchemy is None:
        raise ImportError('Using URI string without sqlalchemy installed.')
    if sqlalchemy is not None and isinstance(con, (str, sqlalchemy.engine.Connectable)):
        return SQLDatabase(con, schema, need_transaction)
    adbc = import_optional_dependency('adbc_driver_manager.dbapi', errors='ignore')
    if adbc and isinstance(con, adbc.Connection):
        return ADBCDatabase(con)
    warnings.warn('pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.', UserWarning, stacklevel=find_stack_level())
    return SQLiteDatabase(con)