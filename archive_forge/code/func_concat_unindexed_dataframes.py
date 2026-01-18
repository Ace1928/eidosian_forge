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
def concat_unindexed_dataframes(dfs, ignore_order=False, **kwargs):
    name = 'concat-' + tokenize(*dfs)
    dsk = {(name, i): (concat_and_check, [(df._name, i) for df in dfs], ignore_order) for i in range(dfs[0].npartitions)}
    kwargs.update({'ignore_order': ignore_order})
    meta = methods.concat([df._meta for df in dfs], axis=1, **kwargs)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=dfs)
    return new_dd_object(graph, name, meta, dfs[0].divisions)