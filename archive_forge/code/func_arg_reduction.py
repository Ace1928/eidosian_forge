from __future__ import annotations
import builtins
import contextlib
import math
import operator
import warnings
from collections.abc import Iterable
from functools import partial
from itertools import product, repeat
from numbers import Integral, Number
import numpy as np
from tlz import accumulate, compose, drop, get, partition_all, pluck
from dask import config
from dask.array import chunk
from dask.array.blockwise import blockwise
from dask.array.core import (
from dask.array.creation import arange, diagonal
from dask.array.dispatch import divide_lookup, nannumel_lookup, numel_lookup
from dask.array.numpy_compat import ComplexWarning
from dask.array.utils import (
from dask.array.wrap import ones, zeros
from dask.base import tokenize
from dask.blockwise import lol_tuples
from dask.highlevelgraph import HighLevelGraph
from dask.utils import (
def arg_reduction(x, chunk, combine, agg, axis=None, keepdims=False, split_every=None, out=None):
    """Generic function for argreduction.

    Parameters
    ----------
    x : Array
    chunk : callable
        Partialed ``arg_chunk``.
    combine : callable
        Partialed ``arg_combine``.
    agg : callable
        Partialed ``arg_agg``.
    axis : int, optional
    split_every : int or dict, optional
    """
    if axis is None:
        axis = tuple(range(x.ndim))
        ravel = True
    elif isinstance(axis, Integral):
        axis = validate_axis(axis, x.ndim)
        axis = (axis,)
        ravel = x.ndim == 1
    else:
        raise TypeError(f"axis must be either `None` or int, got '{axis}'")
    for ax in axis:
        chunks = x.chunks[ax]
        if len(chunks) > 1 and np.isnan(chunks).any():
            raise ValueError('Arg-reductions do not work with arrays that have unknown chunksizes. At some point in your computation this array lost chunking information.\n\nA possible solution is with \n  x.compute_chunk_sizes()')
    name = f'arg-reduce-{tokenize(axis, x, chunk, combine, split_every)}'
    old = x.name
    keys = list(product(*map(range, x.numblocks)))
    offsets = list(product(*(accumulate(operator.add, bd[:-1], 0) for bd in x.chunks)))
    if ravel:
        offset_info = zip(offsets, repeat(x.shape))
    else:
        offset_info = pluck(axis[0], offsets)
    chunks = tuple(((1,) * len(c) if i in axis else c for i, c in enumerate(x.chunks)))
    dsk = {(name,) + k: (chunk, (old,) + k, axis, off) for k, off in zip(keys, offset_info)}
    dtype = np.argmin(asarray_safe([1], like=meta_from_array(x)))
    meta = None
    if is_arraylike(dtype):
        meta = dtype
        dtype = meta.dtype
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[x])
    tmp = Array(graph, name, chunks, dtype=dtype, meta=meta)
    result = _tree_reduce(tmp, agg, axis, keepdims=keepdims, dtype=dtype, split_every=split_every, combine=combine)
    return handle_out(out, result)