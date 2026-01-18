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
def get_unsorted_columns(frames):
    """
    Determine the unsorted column order.

    This should match the output of concat([frames], sort=False)
    """
    new_columns = pd.concat([frame._meta for frame in frames]).columns
    order = []
    for frame in frames:
        order.append(new_columns.get_indexer_for(frame.columns))
    order = np.concatenate(order)
    order = pd.unique(order)
    order = new_columns.take(order)
    return order