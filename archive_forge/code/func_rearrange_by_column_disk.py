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
def rearrange_by_column_disk(df, column, npartitions=None, compute=False):
    """Shuffle using local disk

    See Also
    --------
    rearrange_by_column_tasks:
        Same function, but using tasks rather than partd
        Has a more informative docstring
    """
    if npartitions is None:
        npartitions = df.npartitions
    token = tokenize(df, column, npartitions)
    always_new_token = uuid.uuid1().hex
    p = ('zpartd-' + always_new_token,)
    encode_cls = partd_encode_dispatch(df._meta)
    dsk1 = {p: (maybe_buffered_partd(encode_cls=encode_cls),)}
    name = 'shuffle-partition-' + always_new_token
    dsk2 = {(name, i): (shuffle_group_3, key, column, npartitions, p) for i, key in enumerate(df.__dask_keys__())}
    dependencies = []
    if compute:
        graph = HighLevelGraph.merge(df.dask, dsk1, dsk2)
        graph = HighLevelGraph.from_collections(name, graph, dependencies=[df])
        keys = [p, sorted(dsk2)]
        pp, values = compute_as_if_collection(DataFrame, graph, keys)
        dsk1 = {p: pp}
        dsk2 = dict(zip(sorted(dsk2), values))
    else:
        dependencies.append(df)
    barrier_token = 'barrier-' + always_new_token
    dsk3 = {barrier_token: (barrier, list(dsk2))}
    name = 'shuffle-collect-' + token
    dsk4 = {(name, i): (collect, p, i, df._meta, barrier_token) for i in range(npartitions)}
    divisions = (None,) * (npartitions + 1)
    layer = toolz.merge(dsk1, dsk2, dsk3, dsk4)
    graph = HighLevelGraph.from_collections(name, layer, dependencies=dependencies)
    return new_dd_object(graph, name, df._meta, divisions)