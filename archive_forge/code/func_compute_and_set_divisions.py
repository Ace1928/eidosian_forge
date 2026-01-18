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
def compute_and_set_divisions(df: DataFrame, **kwargs) -> DataFrame:
    mins, maxes, lens = _compute_partition_stats(df.index, allow_overlap=True, **kwargs)
    if len(mins) == len(df.divisions) - 1:
        df._divisions = tuple(mins) + (maxes[-1],)
        if not any((mins[i] >= maxes[i - 1] for i in range(1, len(mins)))):
            return df
    return fix_overlap(df, mins, maxes, lens)