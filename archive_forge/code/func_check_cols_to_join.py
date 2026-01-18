from __future__ import annotations
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType
from pandas.core.dtypes.common import _get_dtype, is_string_dtype
from pyarrow.types import is_dictionary
from modin.pandas.indexing import is_range_like
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
def check_cols_to_join(what, df, col_names):
    """
    Check the frame columns.

    Check if the frame (`df`) has the specified columns (`col_names`). The names referring to
    the index columns are replaced with the actual index column names.

    Parameters
    ----------
    what : str
        Attribute name.
    df : HdkOnNativeDataframe
        The dataframe.
    col_names : list of str
        The column names to check.

    Returns
    -------
    Tuple[HdkOnNativeDataframe, list]
        The aligned data frame and column names.
    """
    cols = df.columns
    new_col_names = col_names
    for i, col in enumerate(col_names):
        if col in cols:
            continue
        new_name = None
        if df._index_cols is not None:
            for c in df._index_cols:
                if col == ColNameCodec.demangle_index_name(c):
                    new_name = c
                    break
        elif df.has_index_cache:
            new_name = f'__index__{0}_{col}'
            df = df._maybe_materialize_rowid()
        if new_name is None:
            raise ValueError(f"'{what}' references unknown column {col}")
        if new_col_names is col_names:
            new_col_names = col_names.copy()
        new_col_names[i] = new_name
    return (df, new_col_names)