from __future__ import annotations
import contextlib
import copy
import math
import re
import types
from enum import Enum, EnumMeta, auto
from typing import (
from typing_extensions import TypeAlias, TypeGuard
import streamlit as st
from streamlit import config, errors
from streamlit import logger as _logger
from streamlit import string_util
from streamlit.errors import StreamlitAPIException
def convert_df_to_data_format(df: DataFrame, data_format: DataFormat) -> DataFrame | Series[Any] | pa.Table | np.ndarray[Any, np.dtype[Any]] | tuple[Any] | list[Any] | set[Any] | dict[str, Any]:
    """Convert a dataframe to the specified data format.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to convert.

    data_format : DataFormat
        The data format to convert to.

    Returns
    -------
    pd.DataFrame, pd.Series, pyarrow.Table, np.ndarray, list, set, tuple, or dict.
        The converted dataframe.
    """
    if data_format in [DataFormat.EMPTY, DataFormat.PANDAS_DATAFRAME, DataFormat.SNOWPARK_OBJECT, DataFormat.PYSPARK_OBJECT, DataFormat.PANDAS_INDEX, DataFormat.PANDAS_STYLER]:
        return df
    elif data_format == DataFormat.NUMPY_LIST:
        import numpy as np
        return np.ndarray(0) if df.empty else df.iloc[:, 0].to_numpy()
    elif data_format == DataFormat.NUMPY_MATRIX:
        import numpy as np
        return np.ndarray(0) if df.empty else df.to_numpy()
    elif data_format == DataFormat.PYARROW_TABLE:
        import pyarrow as pa
        return pa.Table.from_pandas(df)
    elif data_format == DataFormat.PANDAS_SERIES:
        if len(df.columns) != 1:
            raise ValueError(f'DataFrame is expected to have a single column but has {len(df.columns)}.')
        return df[df.columns[0]]
    elif data_format == DataFormat.LIST_OF_RECORDS:
        return _unify_missing_values(df).to_dict(orient='records')
    elif data_format == DataFormat.LIST_OF_ROWS:
        return _unify_missing_values(df).to_numpy().tolist()
    elif data_format == DataFormat.COLUMN_INDEX_MAPPING:
        return _unify_missing_values(df).to_dict(orient='dict')
    elif data_format == DataFormat.COLUMN_VALUE_MAPPING:
        return _unify_missing_values(df).to_dict(orient='list')
    elif data_format == DataFormat.COLUMN_SERIES_MAPPING:
        return df.to_dict(orient='series')
    elif data_format in [DataFormat.LIST_OF_VALUES, DataFormat.TUPLE_OF_VALUES, DataFormat.SET_OF_VALUES]:
        df = _unify_missing_values(df)
        return_list = []
        if len(df.columns) == 1:
            return_list = df[df.columns[0]].tolist()
        elif len(df.columns) >= 1:
            raise ValueError(f'DataFrame is expected to have a single column but has {len(df.columns)}.')
        if data_format == DataFormat.TUPLE_OF_VALUES:
            return tuple(return_list)
        if data_format == DataFormat.SET_OF_VALUES:
            return set(return_list)
        return return_list
    elif data_format == DataFormat.KEY_VALUE_DICT:
        df = _unify_missing_values(df)
        return dict() if df.empty else df.iloc[:, 0].to_dict()
    raise ValueError(f'Unsupported input data format: {data_format}')