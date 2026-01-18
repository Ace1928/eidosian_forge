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
@_deprecated_kwarg('shuffle', 'shuffle_method')
def rearrange_by_column(df, col, npartitions=None, max_branch=None, shuffle_method=None, compute=None, ignore_index=False):
    shuffle_method = shuffle_method or get_default_shuffle_method()
    if shuffle_method != 'p2p' and npartitions is not None and (npartitions < df.npartitions):
        df = df.repartition(npartitions=npartitions)
    if shuffle_method == 'disk':
        return rearrange_by_column_disk(df, col, npartitions, compute=compute)
    elif shuffle_method == 'tasks':
        df2 = rearrange_by_column_tasks(df, col, max_branch, npartitions, ignore_index=ignore_index)
        if ignore_index:
            df2._meta = df2._meta.reset_index(drop=True)
        return df2
    elif shuffle_method == 'p2p':
        from distributed.shuffle import rearrange_by_column_p2p
        return rearrange_by_column_p2p(df, col, npartitions)
    else:
        raise NotImplementedError('Unknown shuffle method %s' % shuffle_method)