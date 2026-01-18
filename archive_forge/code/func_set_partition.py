from __future__ import annotations
import contextlib
import logging
import math
import shutil
import tempfile
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal
import numpy as np
import pandas as pd
import tlz as toolz
from pandas.api.types import is_numeric_dtype
from dask import config
from dask.base import compute, compute_as_if_collection, is_dask_collection, tokenize
from dask.dataframe import methods
from dask.dataframe._compat import PANDAS_GE_300
from dask.dataframe.core import (
from dask.dataframe.dispatch import (
from dask.dataframe.utils import UNKNOWN_CATEGORIES
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ShuffleLayer, SimpleShuffleLayer
from dask.sizeof import sizeof
from dask.utils import M, digit, get_default_shuffle_method
@_deprecated_kwarg('shuffle', 'shuffle_method')
def set_partition(df: DataFrame, index: str | Series, divisions: Sequence, max_branch: int=32, drop: bool=True, shuffle_method: str | None=None, compute: bool | None=None) -> DataFrame:
    """Group DataFrame by index

    Sets a new index and partitions data along that index according to
    divisions.  Divisions are often found by computing approximate quantiles.
    The function ``set_index`` will do both of these steps.

    Parameters
    ----------
    df: DataFrame/Series
        Data that we want to re-partition
    index: string or Series
        Column to become the new index
    divisions: list
        Values to form new divisions between partitions
    drop: bool, default True
        Whether to delete columns to be used as the new index
    shuffle_method: str (optional)
        Either 'disk' for an on-disk shuffle or 'tasks' to use the task
        scheduling framework.  Use 'disk' if you are on a single machine
        and 'tasks' if you are on a distributed cluster.
    max_branch: int (optional)
        If using the task-based shuffle, the amount of splitting each
        partition undergoes.  Increase this for fewer copies but more
        scheduler overhead.

    See Also
    --------
    set_index
    shuffle
    partd
    """
    if isinstance(divisions, tuple):
        divisions = list(divisions)
    if not isinstance(index, Series):
        dtype = df[index].dtype
    else:
        dtype = index.dtype
    if pd.isna(divisions).any() and pd.api.types.is_integer_dtype(dtype):
        divisions = df._meta._constructor_sliced(divisions)
    elif isinstance(dtype, pd.CategoricalDtype) and UNKNOWN_CATEGORIES in dtype.categories:
        divisions = df._meta._constructor_sliced(divisions)
    else:
        divisions = df._meta._constructor_sliced(divisions, dtype=dtype)
    meta = df._meta._constructor_sliced([0])
    meta.index = df._meta_nonempty.index[:1]
    if not isinstance(index, Series):
        partitions = df[index].map_partitions(set_partitions_pre, divisions=divisions, meta=meta)
        df2 = df.assign(_partitions=partitions)
    else:
        partitions = index.map_partitions(set_partitions_pre, divisions=divisions, meta=meta)
        df2 = df.assign(_partitions=partitions, _index=index)
    df3 = rearrange_by_column(df2, '_partitions', max_branch=max_branch, npartitions=len(divisions) - 1, shuffle_method=shuffle_method, compute=compute, ignore_index=True)
    if not isinstance(index, Series):
        df4 = df3.map_partitions(set_index_post_scalar, index_name=index, drop=drop, column_dtype=df.columns.dtype)
    else:
        df4 = df3.map_partitions(set_index_post_series, index_name=index.name, drop=drop, column_dtype=df.columns.dtype)
    divisions = methods.tolist(divisions)
    df4.divisions = tuple((i if not pd.isna(i) else np.nan for i in divisions))
    return df4.map_partitions(M.sort_index)