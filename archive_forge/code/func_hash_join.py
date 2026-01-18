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
@_deprecated_kwarg('shuffle', 'shuffle_method')
def hash_join(lhs, left_on, rhs, right_on, how='inner', npartitions=None, suffixes=('_x', '_y'), shuffle_method=None, indicator=False, max_branch=None):
    """Join two DataFrames on particular columns with hash join

    This shuffles both datasets on the joined column and then performs an
    embarrassingly parallel join partition-by-partition

    >>> hash_join(lhs, 'id', rhs, 'id', how='left', npartitions=10)  # doctest: +SKIP
    """
    if shuffle_method is None:
        shuffle_method = get_default_shuffle_method()
    if shuffle_method == 'p2p':
        from distributed.shuffle import hash_join_p2p
        return hash_join_p2p(lhs=lhs, left_on=left_on, rhs=rhs, right_on=right_on, how=how, npartitions=npartitions, suffixes=suffixes, indicator=indicator)
    if npartitions is None:
        npartitions = max(lhs.npartitions, rhs.npartitions)
    lhs2 = shuffle_func(lhs, left_on, npartitions=npartitions, shuffle_method=shuffle_method, max_branch=max_branch)
    rhs2 = shuffle_func(rhs, right_on, npartitions=npartitions, shuffle_method=shuffle_method, max_branch=max_branch)
    if isinstance(left_on, Index):
        left_on = None
        left_index = True
    else:
        left_index = False
    if isinstance(right_on, Index):
        right_on = None
        right_index = True
    else:
        right_index = False
    kwargs = dict(how=how, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, suffixes=suffixes, indicator=indicator)
    _lhs_meta = lhs._meta_nonempty if len(lhs.columns) else lhs._meta
    _rhs_meta = rhs._meta_nonempty if len(rhs.columns) else rhs._meta
    meta = _lhs_meta.merge(_rhs_meta, **kwargs)
    if isinstance(left_on, list):
        left_on = (list, tuple(left_on))
    if isinstance(right_on, list):
        right_on = (list, tuple(right_on))
    kwargs['result_meta'] = meta
    joined = map_partitions(merge_chunk, lhs2, rhs2, meta=meta, enforce_metadata=False, transform_divisions=False, align_dataframes=False, **kwargs)
    return joined