from __future__ import annotations
import ctypes
import re
from typing import Any
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SettingWithCopyError
import pandas as pd
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.utils import (
def protocol_df_chunk_to_pandas(df: DataFrameXchg) -> pd.DataFrame:
    """
    Convert interchange protocol chunk to ``pd.DataFrame``.

    Parameters
    ----------
    df : DataFrameXchg

    Returns
    -------
    pd.DataFrame
    """
    columns: dict[str, Any] = {}
    buffers = []
    for name in df.column_names():
        if not isinstance(name, str):
            raise ValueError(f'Column {name} is not a string')
        if name in columns:
            raise ValueError(f'Column {name} is not unique')
        col = df.get_column_by_name(name)
        dtype = col.dtype[0]
        if dtype in (DtypeKind.INT, DtypeKind.UINT, DtypeKind.FLOAT, DtypeKind.BOOL):
            columns[name], buf = primitive_column_to_ndarray(col)
        elif dtype == DtypeKind.CATEGORICAL:
            columns[name], buf = categorical_column_to_series(col)
        elif dtype == DtypeKind.STRING:
            columns[name], buf = string_column_to_ndarray(col)
        elif dtype == DtypeKind.DATETIME:
            columns[name], buf = datetime_column_to_ndarray(col)
        else:
            raise NotImplementedError(f'Data type {dtype} not handled yet')
        buffers.append(buf)
    pandas_df = pd.DataFrame(columns)
    pandas_df.attrs['_INTERCHANGE_PROTOCOL_BUFFERS'] = buffers
    return pandas_df