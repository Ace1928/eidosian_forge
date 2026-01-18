from __future__ import annotations
import operator
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial, wraps
from numbers import Integral, Number
from operator import getitem
from pprint import pformat
from typing import Any, ClassVar, Literal, cast
import numpy as np
import pandas as pd
from pandas.api.types import (
from tlz import first, merge, partition_all, remove, unique
import dask.array as da
from dask import config, core
from dask.array.core import Array, normalize_arg
from dask.bag import map_partitions as map_bag_partitions
from dask.base import (
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise
from dask.context import globalmethod
from dask.dataframe import methods
from dask.dataframe._compat import (
from dask.dataframe.accessor import CachedAccessor, DatetimeAccessor, StringAccessor
from dask.dataframe.categorical import CategoricalAccessor, categorize
from dask.dataframe.dispatch import (
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import (
from dask.delayed import Delayed, delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameTreeReduction
from dask.typing import Graph, NestedKeys, no_default
from dask.utils import (
from dask.widgets import get_template
def repartition_npartitions(df, npartitions):
    """Repartition dataframe to a smaller number of partitions"""
    new_name = 'repartition-%d-%s' % (npartitions, tokenize(df))
    if df.npartitions == npartitions:
        return df
    elif df.npartitions > npartitions:
        npartitions_ratio = df.npartitions / npartitions
        new_partitions_boundaries = [int(new_partition_index * npartitions_ratio) for new_partition_index in range(npartitions + 1)]
        return _repartition_from_boundaries(df, new_partitions_boundaries, new_name)
    else:
        original_divisions = divisions = pd.Series(df.divisions).drop_duplicates()
        if df.known_divisions and (is_datetime64_any_dtype(divisions.dtype) or is_numeric_dtype(divisions.dtype)):
            if is_datetime64_any_dtype(divisions.dtype):
                divisions = divisions.values.astype('float64')
            if is_series_like(divisions):
                divisions = divisions.values
            n = len(divisions)
            divisions = np.interp(x=np.linspace(0, n, npartitions + 1), xp=np.linspace(0, n, n), fp=divisions)
            if is_datetime64_any_dtype(original_divisions.dtype):
                divisions = methods.tolist(pd.Series(divisions).astype(original_divisions.dtype))
            elif np.issubdtype(original_divisions.dtype, np.integer):
                divisions = divisions.astype(original_divisions.dtype)
            if isinstance(divisions, np.ndarray):
                divisions = divisions.tolist()
            divisions = list(divisions)
            divisions[0] = df.divisions[0]
            divisions[-1] = df.divisions[-1]
            divisions = list(unique(divisions[:-1])) + [divisions[-1]]
            return df.repartition(divisions=divisions)
        else:
            div, mod = divmod(npartitions, df.npartitions)
            nsplits = [div] * df.npartitions
            nsplits[-1] += mod
            return _split_partitions(df, nsplits, new_name)