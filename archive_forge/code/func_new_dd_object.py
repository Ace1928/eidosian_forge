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
def new_dd_object(dsk, name, meta, divisions, parent_meta=None):
    """Generic constructor for dask.dataframe objects.

    Decides the appropriate output class based on the type of `meta` provided.
    """
    if has_parallel_type(meta):
        return get_parallel_type(meta)(dsk, name, meta, divisions)
    elif is_arraylike(meta) and meta.shape:
        import dask.array as da
        chunks = ((np.nan,) * (len(divisions) - 1),) + tuple(((d,) for d in meta.shape[1:]))
        if len(chunks) > 1:
            if isinstance(dsk, HighLevelGraph):
                layer = dsk.layers[name]
            else:
                layer = dsk
            if isinstance(layer, Blockwise):
                layer.new_axes['j'] = chunks[1][0]
                layer.output_indices = layer.output_indices + ('j',)
            else:
                suffix = (0,) * (len(chunks) - 1)
                for i in range(len(chunks[0])):
                    layer[(name, i) + suffix] = layer.pop((name, i))
        return da.Array(dsk, name=name, chunks=chunks, dtype=meta.dtype)
    else:
        return get_parallel_type(meta)(dsk, name, meta, divisions)