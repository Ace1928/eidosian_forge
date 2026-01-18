from __future__ import annotations
import math
import pickle
import warnings
from functools import partial, wraps
import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal
from tlz import merge_sorted, unique
from dask.base import is_dask_collection, tokenize
from dask.dataframe import methods
from dask.dataframe.core import (
from dask.dataframe.dispatch import group_split_dispatch, hash_object_dispatch
from dask.dataframe.io import from_pandas
from dask.dataframe.shuffle import (
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.layers import BroadcastJoinLayer
from dask.utils import M, apply, get_default_shuffle_method
def concat_indexed_dataframes(dfs, axis=0, join='outer', ignore_order=False, **kwargs):
    """Concatenate indexed dataframes together along the index"""
    warn = axis != 0
    kwargs.update({'ignore_order': ignore_order})
    meta = methods.concat([df._meta for df in dfs], axis=axis, join=join, filter_warning=warn, **kwargs)
    empties = [strip_unknown_categories(df._meta) for df in dfs]
    dfs2, divisions, parts = align_partitions(*dfs)
    name = 'concat-indexed-' + tokenize(join, *dfs)
    parts2 = [[df if df is not None else empty for df, empty in zip(part, empties)] for part in parts]
    filter_warning = True
    uniform = False
    dsk = {(name, i): (methods.concat, part, axis, join, uniform, filter_warning, kwargs) for i, part in enumerate(parts2)}
    for df in dfs2:
        dsk.update(df.dask)
    return new_dd_object(dsk, name, meta, divisions)