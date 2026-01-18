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
def ensure_minimum_chunksize(size, chunks):
    """Determine new chunks to ensure that every chunk >= size

    Parameters
    ----------
    size: int
        The maximum size of any chunk.
    chunks: tuple
        Chunks along one axis, e.g. ``(3, 3, 2)``

    Examples
    --------
    >>> ensure_minimum_chunksize(10, (20, 20, 1))
    (20, 11, 10)
    >>> ensure_minimum_chunksize(3, (1, 1, 3))
    (5,)

    See Also
    --------
    overlap
    """
    if size <= min(chunks):
        return chunks
    output = []
    new = 0
    for c in chunks:
        if c < size:
            if new > size + (size - c):
                output.append(new - (size - c))
                new = size
            else:
                new += c
        if new >= size:
            output.append(new)
            new = 0
        if c >= size:
            new += c
    if new >= size:
        output.append(new)
    elif len(output) >= 1:
        output[-1] += new
    else:
        raise ValueError(f'The overlapping depth {size} is larger than your array {sum(chunks)}.')
    return tuple(output)