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
def partitionwise_graph(func, layer_name, *args, **kwargs):
    """
    Apply a function partition-wise across arguments to create layer of a graph

    This applies a function, ``func``, in an embarrassingly parallel fashion
    across partitions/chunks in the provided arguments.  It handles Dataframes,
    Arrays, and scalars smoothly, and relies on the ``blockwise`` machinery
    to provide a nicely symbolic graph.

    It is most commonly used in other graph-building functions to create the
    appropriate layer of the resulting dataframe.

    Parameters
    ----------
    func: callable
    layer_name: str
        Descriptive name for the operation. Used as the output name
        in the resulting ``Blockwise`` graph layer.
    *args:
    **kwargs:

    Returns
    -------
    out: Blockwise graph

    Examples
    --------
    >>> subgraph = partitionwise_graph(function, x, y, z=123)  # doctest: +SKIP
    >>> layer = partitionwise_graph(function, df, x, z=123)  # doctest: +SKIP
    >>> graph = HighLevelGraph.from_collections(name, layer, dependencies=[df, x])  # doctest: +SKIP
    >>> result = new_dd_object(graph, name, metadata, df.divisions)  # doctest: +SKIP

    See Also
    --------
    map_partitions
    """
    pairs = []
    numblocks = {}
    for arg in args:
        if isinstance(arg, _Frame):
            pairs.extend([arg._name, 'i'])
            numblocks[arg._name] = (arg.npartitions,)
        elif isinstance(arg, Scalar):
            pairs.extend([arg._name, 'i'])
            numblocks[arg._name] = (1,)
        elif isinstance(arg, Array):
            if arg.ndim == 1:
                pairs.extend([arg.name, 'i'])
            elif arg.ndim == 0:
                pairs.extend([arg.name, ''])
            elif arg.ndim == 2:
                pairs.extend([arg.name, 'ij'])
            else:
                raise ValueError("Can't add multi-dimensional array to dataframes")
            numblocks[arg._name] = arg.numblocks
        elif isinstance(arg, BlockwiseDep):
            if len(arg.numblocks) == 1:
                pairs.extend([arg, 'i'])
            elif len(arg.numblocks) == 2:
                pairs.extend([arg, 'ij'])
            else:
                raise ValueError(f'BlockwiseDep arg {arg!r} has {len(arg.numblocks)} dimensions; only 1 or 2 are supported.')
        else:
            pairs.extend([arg, None])
    return blockwise(func, layer_name, 'i', *pairs, numblocks=numblocks, concatenate=True, **kwargs)