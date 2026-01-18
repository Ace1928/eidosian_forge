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
def shuffle_group_2(df, cols, ignore_index, nparts):
    if not len(df):
        return ({}, df)
    if isinstance(cols, str):
        cols = [cols]
    if cols and cols[0] == '_partitions':
        ind = df[cols[0]].astype(np.int32)
    else:
        ind = (hash_object_dispatch(df[cols] if cols else df, index=False) % int(nparts)).astype(np.int32)
    n = ind.max() + 1
    result2 = group_split_dispatch(df, ind, n, ignore_index=ignore_index)
    return (result2, df.iloc[:0])