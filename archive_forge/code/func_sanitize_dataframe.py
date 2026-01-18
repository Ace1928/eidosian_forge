from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import json
import itertools
import re
import sys
import traceback
import warnings
from typing import (
from types import ModuleType
import jsonschema
import pandas as pd
import numpy as np
from pandas.api.types import infer_dtype
from altair.utils.schemapi import SchemaBase
from altair.utils._dfi_types import Column, DtypeKind, DataFrame as DfiDataFrame
from typing import Literal, Protocol, TYPE_CHECKING, runtime_checkable
def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize a DataFrame to prepare it for serialization.

    * Make a copy
    * Convert RangeIndex columns to strings
    * Raise ValueError if column names are not strings
    * Raise ValueError if it has a hierarchical index.
    * Convert categoricals to strings.
    * Convert np.bool_ dtypes to Python bool objects
    * Convert np.int dtypes to Python int objects
    * Convert floats to objects and replace NaNs/infs with None.
    * Convert DateTime dtypes into appropriate string representations
    * Convert Nullable integers to objects and replace NaN with None
    * Convert Nullable boolean to objects and replace NaN with None
    * convert dedicated string column to objects and replace NaN with None
    * Raise a ValueError for TimeDelta dtypes
    """
    df = df.copy()
    if isinstance(df.columns, pd.RangeIndex):
        df.columns = df.columns.astype(str)
    for col_name in df.columns:
        if not isinstance(col_name, str):
            raise ValueError('Dataframe contains invalid column name: {0!r}. Column names must be strings'.format(col_name))
    if isinstance(df.index, pd.MultiIndex):
        raise ValueError('Hierarchical indices not supported')
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError('Hierarchical indices not supported')

    def to_list_if_array(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        else:
            return val
    for dtype_item in df.dtypes.items():
        col_name = cast(str, dtype_item[0])
        dtype = dtype_item[1]
        dtype_name = str(dtype)
        if dtype_name == 'category':
            col = df[col_name].astype(object)
            df[col_name] = col.where(col.notnull(), None)
        elif dtype_name == 'string':
            col = df[col_name].astype(object)
            df[col_name] = col.where(col.notnull(), None)
        elif dtype_name == 'bool':
            df[col_name] = df[col_name].astype(object)
        elif dtype_name == 'boolean':
            col = df[col_name].astype(object)
            df[col_name] = col.where(col.notnull(), None)
        elif dtype_name.startswith('datetime') or dtype_name.startswith('timestamp'):
            df[col_name] = df[col_name].apply(lambda x: x.isoformat()).replace('NaT', '')
        elif dtype_name.startswith('timedelta'):
            raise ValueError('Field "{col_name}" has type "{dtype}" which is not supported by Altair. Please convert to either a timestamp or a numerical value.'.format(col_name=col_name, dtype=dtype))
        elif dtype_name.startswith('geometry'):
            continue
        elif dtype_name in {'Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64', 'Float32', 'Float64'}:
            col = df[col_name].astype(object)
            df[col_name] = col.where(col.notnull(), None)
        elif numpy_is_subtype(dtype, np.integer):
            df[col_name] = df[col_name].astype(object)
        elif numpy_is_subtype(dtype, np.floating):
            col = df[col_name]
            bad_values = col.isnull() | np.isinf(col)
            df[col_name] = col.astype(object).where(~bad_values, None)
        elif dtype == object:
            col = df[col_name].astype(object).apply(to_list_if_array)
            df[col_name] = col.where(col.notnull(), None)
    return df