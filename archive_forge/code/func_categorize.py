from __future__ import annotations
from collections import defaultdict
from numbers import Integral
import pandas as pd
from pandas.api.types import is_scalar
from tlz import partition_all
from dask.base import compute_as_if_collection, tokenize
from dask.dataframe import methods
from dask.dataframe.accessor import Accessor
from dask.dataframe.dispatch import (  # noqa: F401
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
def categorize(df, columns=None, index=None, split_every=None, **kwargs):
    """Convert columns of the DataFrame to category dtype.

    Parameters
    ----------
    columns : list, optional
        A list of column names to convert to categoricals. By default any
        column with an object dtype is converted to a categorical, and any
        unknown categoricals are made known.
    index : bool, optional
        Whether to categorize the index. By default, object indices are
        converted to categorical, and unknown categorical indices are made
        known. Set True to always categorize the index, False to never.
    split_every : int, optional
        Group partitions into groups of this size while performing a
        tree-reduction. If set to False, no tree-reduction will be used.
        Default is 16.
    kwargs
        Keyword arguments are passed on to compute.
    """
    meta = df._meta
    if columns is None:
        columns = list(meta.select_dtypes(['object', 'string', 'category']).columns)
    elif is_scalar(columns):
        columns = [columns]
    columns = [c for c in columns if not (is_categorical_dtype(meta[c]) and has_known_categories(meta[c]))]
    if index is not False:
        if is_categorical_dtype(meta.index):
            index = not has_known_categories(meta.index)
        elif index is None:
            index = str(meta.index.dtype) in ('object', 'string')
    if not len(columns) and index is False:
        return df
    if split_every is None:
        split_every = 16
    elif split_every is False:
        split_every = df.npartitions
    elif not isinstance(split_every, Integral) or split_every < 2:
        raise ValueError('split_every must be an integer >= 2')
    token = tokenize(df, columns, index, split_every)
    a = 'get-categories-chunk-' + token
    dsk = {(a, i): (_get_categories, key, columns, index) for i, key in enumerate(df.__dask_keys__())}
    prefix = 'get-categories-agg-' + token
    k = df.npartitions
    depth = 0
    while k > split_every:
        b = prefix + str(depth)
        for part_i, inds in enumerate(partition_all(split_every, range(k))):
            dsk[b, part_i] = (_get_categories_agg, [(a, i) for i in inds])
        k = part_i + 1
        a = b
        depth += 1
    dsk[prefix, 0] = (_get_categories_agg, [(a, i) for i in range(k)])
    graph = HighLevelGraph.from_collections(prefix, dsk, dependencies=[df])
    categories, index = compute_as_if_collection(df.__class__, graph, (prefix, 0), **kwargs)
    categories = {k: v.sort_values() for k, v in categories.items()}
    return df.map_partitions(_categorize_block, categories, index)