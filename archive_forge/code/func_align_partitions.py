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
def align_partitions(*dfs):
    """Mutually partition and align DataFrame blocks

    This serves as precursor to multi-dataframe operations like join, concat,
    or merge.

    Parameters
    ----------
    dfs: sequence of dd.DataFrame, dd.Series and dd.base.Scalar
        Sequence of dataframes to be aligned on their index

    Returns
    -------
    dfs: sequence of dd.DataFrame, dd.Series and dd.base.Scalar
        These must have consistent divisions with each other
    divisions: tuple
        Full divisions sequence of the entire result
    result: list
        A list of lists of keys that show which data exist on which
        divisions
    """
    _is_broadcastable = partial(is_broadcastable, dfs)
    dfs1 = [df for df in dfs if isinstance(df, _Frame) and (not _is_broadcastable(df))]
    if len(dfs) == 0:
        raise ValueError('dfs contains no DataFrame and Series')
    if not all((df.known_divisions for df in dfs1)):
        raise ValueError("Not all divisions are known, can't align partitions. Please use `set_index` to set the index.")
    divisions = list(unique(merge_sorted(*[df.divisions for df in dfs1])))
    if len(divisions) == 1:
        divisions = (divisions[0], divisions[0])
    dfs2 = [df.repartition(divisions, force=True) if isinstance(df, _Frame) else df for df in dfs]
    result = list()
    inds = [0 for df in dfs]
    for d in divisions[:-1]:
        L = list()
        for i, df in enumerate(dfs2):
            if isinstance(df, _Frame):
                j = inds[i]
                divs = df.divisions
                if j < len(divs) - 1 and divs[j] == d:
                    L.append((df._name, inds[i]))
                    inds[i] += 1
                else:
                    L.append(None)
            else:
                L.append(None)
        result.append(L)
    return (dfs2, tuple(divisions), result)