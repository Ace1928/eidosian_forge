from __future__ import annotations
import warnings
from numbers import Integral, Number
import numpy as np
from tlz import concat, get, partial
from tlz.curried import map
from dask.array import chunk
from dask.array.core import Array, concatenate, map_blocks, unify_chunks
from dask.array.creation import empty_like, full_like
from dask.array.numpy_compat import normalize_axis_tuple
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayOverlapLayer
from dask.utils import derived_from
def overlap_internal(x, axes):
    """Share boundaries between neighboring blocks

    Parameters
    ----------

    x: da.Array
        A dask array
    axes: dict
        The size of the shared boundary per axis

    The axes input informs how many cells to overlap between neighboring blocks
    {0: 2, 2: 5} means share two cells in 0 axis, 5 cells in 2 axis
    """
    token = tokenize(x, axes)
    name = 'overlap-' + token
    graph = ArrayOverlapLayer(name=x.name, axes=axes, chunks=x.chunks, numblocks=x.numblocks, token=token)
    graph = HighLevelGraph.from_collections(name, graph, dependencies=[x])
    chunks = _overlap_internal_chunks(x.chunks, axes)
    return Array(graph, name, chunks, meta=x)