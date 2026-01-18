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
def cumreduction(func, binop, ident, x, axis=None, dtype=None, out=None, method='sequential', preop=None):
    """Generic function for cumulative reduction

    Parameters
    ----------
    func: callable
        Cumulative function like np.cumsum or np.cumprod
    binop: callable
        Associated binary operator like ``np.cumsum->add`` or ``np.cumprod->mul``
    ident: Number
        Associated identity like ``np.cumsum->0`` or ``np.cumprod->1``
    x: dask Array
    axis: int
    dtype: dtype
    method : {'sequential', 'blelloch'}, optional
        Choose which method to use to perform the cumsum.  Default is 'sequential'.

        * 'sequential' performs the scan of each prior block before the current block.
        * 'blelloch' is a work-efficient parallel scan.  It exposes parallelism by first
          calling ``preop`` on each block and combines the values via a binary tree.
          This method may be faster or more memory efficient depending on workload,
          scheduler, and hardware.  More benchmarking is necessary.
    preop: callable, optional
        Function used by 'blelloch' method,
        like ``np.cumsum->np.sum`` or ``np.cumprod->np.prod``

    Returns
    -------
    dask array

    See also
    --------
    cumsum
    cumprod
    """
    if method == 'blelloch':
        if preop is None:
            raise TypeError('cumreduction with "blelloch" method required `preop=` argument')
        return prefixscan_blelloch(func, preop, binop, x, axis, dtype, out=out)
    elif method != 'sequential':
        raise ValueError(f'Invalid method for cumreduction.  Expected "sequential" or "blelloch".  Got: {method!r}')
    if axis is None:
        x = x.flatten().rechunk(chunks=x.npartitions)
        axis = 0
    if dtype is None:
        dtype = getattr(func(np.ones((0,), dtype=x.dtype)), 'dtype', object)
    assert isinstance(axis, Integral)
    axis = validate_axis(axis, x.ndim)
    m = x.map_blocks(func, axis=axis, dtype=dtype)
    name = f'{func.__name__}-{tokenize(func, axis, binop, ident, x, dtype)}'
    n = x.numblocks[axis]
    full = slice(None, None, None)
    slc = (full,) * axis + (slice(-1, None),) + (full,) * (x.ndim - axis - 1)
    indices = list(product(*[range(nb) if i != axis else [0] for i, nb in enumerate(x.numblocks)]))
    dsk = dict()
    for ind in indices:
        shape = tuple((x.chunks[i][ii] if i != axis else 1 for i, ii in enumerate(ind)))
        dsk[(name, 'extra') + ind] = (apply, np.full_like, (x._meta, ident, m.dtype), {'shape': shape})
        dsk[(name,) + ind] = (m.name,) + ind
    for i in range(1, n):
        last_indices = indices
        indices = list(product(*[range(nb) if ii != axis else [i] for ii, nb in enumerate(x.numblocks)]))
        for old, ind in zip(last_indices, indices):
            this_slice = (name, 'extra') + ind
            dsk[this_slice] = (binop, (name, 'extra') + old, (operator.getitem, (m.name,) + old, slc))
            dsk[(name,) + ind] = (binop, this_slice, (m.name,) + ind)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[m])
    result = Array(graph, name, x.chunks, m.dtype, meta=x._meta)
    return handle_out(out, result)