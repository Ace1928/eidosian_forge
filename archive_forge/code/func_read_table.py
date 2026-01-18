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
def read_table(self, table_name: str, index_col: str | list[str] | None=None, coerce_float: bool=True, parse_dates=None, columns=None, schema: str | None=None, chunksize: int | None=None, dtype_backend: DtypeBackend | Literal['numpy']='numpy') -> DataFrame | Iterator[DataFrame]:
    """
        Read SQL database table into a DataFrame.

        Parameters
        ----------
        table_name : str
            Name of SQL table in database.
        coerce_float : bool, default True
            Raises NotImplementedError
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg}``, where the arg corresponds
              to the keyword arguments of :func:`pandas.to_datetime`.
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, default: None
            List of column names to select from SQL table.
        schema : string, default None
            Name of SQL schema in database to query (if database flavor
            supports this).  If specified, this overwrites the default
            schema of the SQL database object.
        chunksize : int, default None
            Raises NotImplementedError
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
        DataFrame

        See Also
        --------
        pandas.read_sql_table
        SQLDatabase.read_query

        """
    if coerce_float is not True:
        raise NotImplementedError("'coerce_float' is not implemented for ADBC drivers")
    if chunksize:
        raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")
    if columns:
        if index_col:
            index_select = maybe_make_list(index_col)
        else:
            index_select = []
        to_select = index_select + columns
        select_list = ', '.join((f'"{x}"' for x in to_select))
    else:
        select_list = '*'
    if schema:
        stmt = f'SELECT {select_list} FROM {schema}.{table_name}'
    else:
        stmt = f'SELECT {select_list} FROM {table_name}'
    mapping: type[ArrowDtype] | None | Callable
    if dtype_backend == 'pyarrow':
        mapping = ArrowDtype
    elif dtype_backend == 'numpy_nullable':
        from pandas.io._util import _arrow_dtype_mapping
        mapping = _arrow_dtype_mapping().get
    elif using_pyarrow_string_dtype():
        from pandas.io._util import arrow_string_types_mapper
        arrow_string_types_mapper()
    else:
        mapping = None
    with self.con.cursor() as cur:
        cur.execute(stmt)
        df = cur.fetch_arrow_table().to_pandas(types_mapper=mapping)
    return _wrap_result_adbc(df, index_col=index_col, parse_dates=parse_dates)