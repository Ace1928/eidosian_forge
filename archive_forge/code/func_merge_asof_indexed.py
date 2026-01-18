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
def merge_asof_indexed(left, right, **kwargs):
    dsk = dict()
    name = 'asof-join-indexed-' + tokenize(left, right, **kwargs)
    meta = pd.merge_asof(left._meta_nonempty, right._meta_nonempty, **kwargs)
    if all(map(pd.isnull, left.divisions)):
        return from_pandas(meta.iloc[len(meta):], npartitions=left.npartitions)
    if all(map(pd.isnull, right.divisions)):
        return map_partitions(pd.merge_asof, left, right=right, left_index=True, right_index=True, meta=meta)
    dependencies = [left, right]
    tails = heads = None
    if kwargs['direction'] in ['backward', 'nearest']:
        tails = compute_tails(right, by=kwargs['right_by'])
        dependencies.append(tails)
    if kwargs['direction'] in ['forward', 'nearest']:
        heads = compute_heads(right, by=kwargs['right_by'])
        dependencies.append(heads)
    for i, J in enumerate(pair_partitions(left.divisions, right.divisions)):
        frames = []
        for j, lower, upper in J:
            slice = (methods.boundary_slice, (left._name, i), lower, upper, False)
            tail = (tails._name, j) if tails is not None else None
            head = (heads._name, j) if heads is not None else None
            frames.append((apply, merge_asof_padded, [slice, (right._name, j), tail, head], kwargs))
        dsk[name, i] = (methods.concat, frames)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=dependencies)
    result = new_dd_object(graph, name, meta, left.divisions)
    return result