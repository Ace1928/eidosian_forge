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
def rearrange_by_column_tasks(df, column, max_branch=32, npartitions=None, ignore_index=False):
    """Order divisions of DataFrame so that all values within column(s) align

    This enacts a task-based shuffle.  It contains most of the tricky logic
    around the complex network of tasks.  Typically before this function is
    called a new column, ``"_partitions"`` has been added to the dataframe,
    containing the output partition number of every row.  This function
    produces a new dataframe where every row is in the proper partition.  It
    accomplishes this by splitting each input partition into several pieces,
    and then concatenating pieces from different input partitions into output
    partitions.  If there are enough partitions then it does this work in
    stages to avoid scheduling overhead.

    Lets explain the motivation for this further.  Imagine that we have 1000
    input partitions and 1000 output partitions. In theory we could split each
    input into 1000 pieces, and then move the 1 000 000 resulting pieces
    around, and then concatenate them all into 1000 output groups.  This would
    be fine, but the central scheduling overhead of 1 000 000 tasks would
    become a bottleneck.  Instead we do this in stages so that we split each of
    the 1000 inputs into 30 pieces (we now have 30 000 pieces) move those
    around, concatenate back down to 1000, and then do the same process again.
    This has the same result as the full transfer, but now we've moved data
    twice (expensive) but done so with only 60 000 tasks (cheap).

    Note that the `column` input may correspond to a list of columns (rather
    than just a single column name).  In this case, the `shuffle_group` and
    `shuffle_group_2` functions will use hashing to map each row to an output
    partition. This approach may require the same rows to be hased multiple
    times, but avoids the need to assign a new "_partitions" column.

    Parameters
    ----------
    df: dask.dataframe.DataFrame
    column: str or list
        A column name on which we want to split, commonly ``"_partitions"``
        which is assigned by functions upstream.  This could also be a list of
        columns (in which case shuffle_group will create a hash array/column).
    max_branch: int
        The maximum number of splits per input partition.  Defaults to 32.
        If there are more partitions than this then the shuffling will occur in
        stages in order to avoid creating npartitions**2 tasks
        Increasing this number increases scheduling overhead but decreases the
        number of full-dataset transfers that we have to make.
    npartitions: Optional[int]
        The desired number of output partitions

    Returns
    -------
    df3: dask.dataframe.DataFrame

    See also
    --------
    rearrange_by_column_disk: same operation, but uses partd
    rearrange_by_column: parent function that calls this or rearrange_by_column_disk
    shuffle_group: does the actual splitting per-partition
    """
    max_branch = max_branch or 32
    if (npartitions or df.npartitions) <= max_branch:
        token = tokenize(df, column, npartitions)
        shuffle_name = f'simple-shuffle-{token}'
        npartitions = npartitions or df.npartitions
        shuffle_layer = SimpleShuffleLayer(shuffle_name, column, npartitions, df.npartitions, ignore_index, df._name, df._meta)
        graph = HighLevelGraph.from_collections(shuffle_name, shuffle_layer, dependencies=[df])
        return new_dd_object(graph, shuffle_name, df._meta, [None] * (npartitions + 1))
    n = df.npartitions
    stages = int(math.ceil(math.log(n) / math.log(max_branch)))
    if stages > 1:
        k = int(math.ceil(n ** (1 / stages)))
    else:
        k = n
    inputs = [tuple((digit(i, j, k) for j in range(stages))) for i in range(k ** stages)]
    npartitions_orig = df.npartitions
    token = tokenize(df, stages, column, n, k)
    for stage in range(stages):
        stage_name = f'shuffle-{stage}-{token}'
        stage_layer = ShuffleLayer(stage_name, column, inputs, stage, npartitions, n, k, ignore_index, df._name, df._meta)
        graph = HighLevelGraph.from_collections(stage_name, stage_layer, dependencies=[df])
        df = new_dd_object(graph, stage_name, df._meta, df.divisions)
    if npartitions is not None and npartitions != npartitions_orig:
        token = tokenize(df, npartitions)
        repartition_group_token = 'repartition-group-' + token
        dsk = {(repartition_group_token, i): (shuffle_group_2, k, column, ignore_index, npartitions) for i, k in enumerate(df.__dask_keys__())}
        repartition_get_name = 'repartition-get-' + token
        for p in range(npartitions):
            dsk[repartition_get_name, p] = (shuffle_group_get, (repartition_group_token, p % npartitions_orig), p)
        graph2 = HighLevelGraph.from_collections(repartition_get_name, dsk, dependencies=[df])
        df2 = new_dd_object(graph2, repartition_get_name, df._meta, [None] * (npartitions + 1))
    else:
        df2 = df
        df2.divisions = (None,) * (npartitions_orig + 1)
    return df2