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
def get_data_for_join_by_index(left, right, how, left_on, right_on, sort, suffixes):
    """
    Return the column names, dtypes and expres, required for join by index.

    This is a helper function, used by `HdkOnNativeDataframe.join()`, when joining by index.

    Parameters
    ----------
    left : HdkOnNativeDataframe
        A frame to join.
    right : HdkOnNativeDataframe
        A frame to join with.
    how : str
        A type of join.
    left_on : list of str
        A list of columns for the left frame to join on.
    right_on : list of str
        A list of columns for the right frame to join on.
    sort : bool
        Sort the result by join keys.
    suffixes : list-like of str
        A length-2 sequence of suffixes to add to overlapping column names
        of left and right operands respectively.

    Returns
    -------
    tuple

    The index columns, exprs, dtypes and columns.
    """

    def to_empty_pandas_df(df):
        idx = df._index_cache.get() if df.has_index_cache else None
        if idx is not None:
            idx = idx[:1]
        elif df._index_cols is not None:
            if len(df._index_cols) > 1:
                arrays = [[i] for i in range(len(df._index_cols))]
                names = [ColNameCodec.demangle_index_name(n) for n in df._index_cols]
                idx = pandas.MultiIndex.from_arrays(arrays, names=names)
            else:
                idx = pandas.Index(name=ColNameCodec.demangle_index_name(df._index_cols[0]))
        return pandas.DataFrame(columns=df.columns, index=idx)
    new_dtypes = []
    exprs = {}
    merged = to_empty_pandas_df(left).merge(to_empty_pandas_df(right), how=how, left_on=left_on, right_on=right_on, sort=sort, suffixes=suffixes)
    if len(merged.index.names) == 1 and merged.index.names[0] is None:
        index_cols = None
    else:
        index_cols = ColNameCodec.mangle_index_names(merged.index.names)
        for name in index_cols:
            df = left if name in left._dtypes else right
            exprs[name] = df.ref(name)
            new_dtypes.append(df._dtypes[name])
    left_col_names = set(left.columns)
    right_col_names = set(right.columns)
    for col in merged.columns:
        orig_name = col
        if orig_name in left_col_names:
            df = left
        elif orig_name in right_col_names:
            df = right
        elif suffixes is None:
            raise ValueError(f'Unknown column {col}')
        elif col.endswith(suffixes[0]) and (orig_name := col[0:-len(suffixes[0])]) in left_col_names and (orig_name in right_col_names):
            df = left
        elif col.endswith(suffixes[1]) and (orig_name := col[0:-len(suffixes[1])]) in right_col_names and (orig_name in left_col_names):
            df = right
        else:
            raise ValueError(f'Unknown column {col}')
        exprs[col] = df.ref(orig_name)
        new_dtypes.append(df._dtypes[orig_name])
    return (index_cols, exprs, new_dtypes, merged.columns)