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
def set_sorted_index(df: DataFrame, index: str | Series, drop: bool=True, divisions: Sequence | None=None, **kwargs) -> DataFrame:
    if isinstance(index, Series):
        meta = df._meta.set_index(index._meta, drop=drop)
    else:
        meta = df._meta.set_index(index, drop=drop)
    result = map_partitions(M.set_index, df, index, drop=drop, meta=meta, align_dataframes=False, transform_divisions=False)
    if not divisions:
        return compute_and_set_divisions(result, **kwargs)
    elif len(divisions) != len(df.divisions):
        msg = 'When doing `df.set_index(col, sorted=True, divisions=...)`, divisions indicates known splits in the index column. In this case divisions must be the same length as the existing divisions in `df`\n\nIf the intent is to repartition into new divisions after setting the index, you probably want:\n\n`df.set_index(col, sorted=True).repartition(divisions=divisions)`'
        raise ValueError(msg)
    result.divisions = tuple(divisions)
    return result