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
def partitioning_index(df, npartitions, cast_dtype=None):
    """
    Computes a deterministic index mapping each record to a partition.

    Identical rows are mapped to the same partition.

    Parameters
    ----------
    df : DataFrame/Series/Index
    npartitions : int
        The number of partitions to group into.
    cast_dtype : dtype, optional
        The dtype to cast to to avoid nullability issues

    Returns
    -------
    partitions : ndarray
        An array of int64 values mapping each record to a partition.
    """
    if cast_dtype is not None:
        df = df.astype(cast_dtype, errors='ignore')
    res = hash_object_dispatch(df, index=False) % int(npartitions)
    return res.astype(np.min_scalar_type(-(npartitions - 1)))