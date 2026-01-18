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
def elemwise(op, *args, meta=no_default, out=None, transform_divisions=True, **kwargs):
    """Elementwise operation for Dask dataframes

    Parameters
    ----------
    op: callable
        Function to apply across input dataframes
    *args: DataFrames, Series, Scalars, Arrays, etc.
        The arguments of the operation
    meta: pd.DataFrame, pd.Series (optional)
        Valid metadata for the operation.  Will evaluate on a small piece of
        data if not provided.
    transform_divisions: boolean
        If the input is a ``dask.dataframe.Index`` we normally will also apply
        the function onto the divisions and apply those transformed divisions
        to the output.  You can pass ``transform_divisions=False`` to override
        this behavior
    out : dask.DataFrame, dask.Series, dask.Scalar, or None
        If out is a dask.DataFrame, dask.Series or dask.Scalar then
        this overwrites the contents of it with the result
    **kwargs: scalars

    Examples
    --------
    >>> elemwise(operator.add, df.x, df.y)  # doctest: +SKIP
    """
    _name = funcname(op) + '-' + tokenize(op, *args, **kwargs)
    args = _maybe_from_pandas(args)
    from dask.dataframe.multi import _maybe_align_partitions
    args = _maybe_align_partitions(args)
    dasks = [arg for arg in args if isinstance(arg, (_Frame, Scalar, Array))]
    dfs = [df for df in dasks if isinstance(df, _Frame)]
    deps = dasks.copy()
    for i, a in enumerate(dasks):
        if not isinstance(a, Array):
            continue
        if not all((not a.chunks or len(a.chunks[0]) == df.npartitions for df in dfs)):
            msg = 'When combining dask arrays with dataframes they must match chunking exactly.  Operation: %s' % funcname(op)
            raise ValueError(msg)
        if a.ndim > 1:
            a = a.rechunk({i + 1: d for i, d in enumerate(a.shape[1:])})
            dasks[i] = a
    divisions = dfs[0].divisions
    if transform_divisions and isinstance(dfs[0], Index) and (len(dfs) == 1):
        try:
            divisions = op(*[pd.Index(arg.divisions) if arg is dfs[0] else arg for arg in args], **kwargs)
            if isinstance(divisions, pd.Index):
                divisions = methods.tolist(divisions)
        except Exception:
            pass
        else:
            if not valid_divisions(divisions):
                divisions = [None] * (dfs[0].npartitions + 1)
    _is_broadcastable = partial(is_broadcastable, dfs)
    dfs = list(remove(_is_broadcastable, dfs))
    other = [(i, arg) for i, arg in enumerate(args) if not isinstance(arg, (_Frame, Scalar, Array))]
    dsk = partitionwise_graph(op, _name, *args, **kwargs)
    graph = HighLevelGraph.from_collections(_name, dsk, dependencies=deps)
    if meta is no_default:
        if len(dfs) >= 2 and (not all((hasattr(d, 'npartitions') for d in dasks))):
            msg = 'elemwise with 2 or more DataFrames and Scalar is not supported'
            raise NotImplementedError(msg)
        parts = [d._meta if _is_broadcastable(d) else np.empty((), dtype=d.dtype) if isinstance(d, Array) else d._meta_nonempty for d in dasks]
        with raise_on_meta_error(funcname(op)):
            meta = partial_by_order(*parts, function=op, other=other)
    result = new_dd_object(graph, _name, meta, divisions)
    return handle_out(out, result)