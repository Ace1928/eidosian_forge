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
def from_dataframe(df, allow_copy: bool=True) -> pd.DataFrame:
    """
    Build a ``pd.DataFrame`` from any DataFrame supporting the interchange protocol.

    Parameters
    ----------
    df : DataFrameXchg
        Object supporting the interchange protocol, i.e. `__dataframe__` method.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    >>> df_not_necessarily_pandas = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> interchange_object = df_not_necessarily_pandas.__dataframe__()
    >>> interchange_object.column_names()
    Index(['A', 'B'], dtype='object')
    >>> df_pandas = (pd.api.interchange.from_dataframe
    ...              (interchange_object.select_columns_by_name(['A'])))
    >>> df_pandas
         A
    0    1
    1    2

    These methods (``column_names``, ``select_columns_by_name``) should work
    for any dataframe library which implements the interchange protocol.
    """
    if isinstance(df, pd.DataFrame):
        return df
    if not hasattr(df, '__dataframe__'):
        raise ValueError('`df` does not support __dataframe__')
    return _from_dataframe(df.__dataframe__(allow_copy=allow_copy), allow_copy=allow_copy)