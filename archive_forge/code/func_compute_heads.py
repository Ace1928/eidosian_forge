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
def compute_heads(ddf, by=None):
    """For each partition, returns the first row of the next nonempty
    partition.
    """
    empty = ddf._meta.iloc[0:0]
    if by is None:
        return suffix_reduction(most_recent_head, ddf, empty)
    else:
        kwargs = {'by': by}
        return suffix_reduction(most_recent_head_summary, ddf, empty, **kwargs)