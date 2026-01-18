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
def read_sql_table(table_name: str, con, schema: str | None=None, index_col: str | list[str] | None=None, coerce_float: bool=True, parse_dates: list[str] | dict[str, str] | None=None, columns: list[str] | None=None, chunksize: int | None=None, dtype_backend: DtypeBackend | lib.NoDefault=lib.no_default) -> DataFrame | Iterator[DataFrame]:
    """
    Read SQL database table into a DataFrame.

    Given a table name and a SQLAlchemy connectable, returns a DataFrame.
    This function does not support DBAPI connections.

    Parameters
    ----------
    table_name : str
        Name of SQL table in database.
    con : SQLAlchemy connectable or str
        A database URI could be provided as str.
        SQLite DBAPI connection mode not supported.
    schema : str, default None
        Name of SQL schema in database to query (if database flavor
        supports this). Uses default schema if None (default).
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Can result in loss of Precision.
    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default None
        List of column names to select from SQL table.
    chunksize : int, default None
        If specified, returns an iterator where `chunksize` is the number of
        rows to include in each chunk.
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]
        A SQL table is returned as two-dimensional data structure with labeled
        axes.

    See Also
    --------
    read_sql_query : Read SQL query into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information will be converted to UTC.

    Examples
    --------
    >>> pd.read_sql_table('table_name', 'postgres:///db_name')  # doctest:+SKIP
    """
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = 'numpy'
    assert dtype_backend is not lib.no_default
    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        if not pandas_sql.has_table(table_name):
            raise ValueError(f'Table {table_name} not found')
        table = pandas_sql.read_table(table_name, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, columns=columns, chunksize=chunksize, dtype_backend=dtype_backend)
    if table is not None:
        return table
    else:
        raise ValueError(f'Table {table_name} not found', con)