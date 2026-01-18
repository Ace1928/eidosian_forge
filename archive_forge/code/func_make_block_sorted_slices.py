from __future__ import annotations
import bisect
import functools
import math
import warnings
from itertools import product
from numbers import Integral, Number
from operator import itemgetter
import numpy as np
from tlz import concat, memoize, merge, pluck
from dask import config, core, utils
from dask.array.chunk import getitem
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import cached_cumsum, is_arraylike
def make_block_sorted_slices(index, chunks):
    """Generate blockwise-sorted index pairs for shuffling an array.

    Parameters
    ----------
    index : ndarray
        An array of index positions.
    chunks : tuple
        Chunks from the original dask array

    Returns
    -------
    index2 : ndarray
        Same values as `index`, but each block has been sorted
    index3 : ndarray
        The location of the values of `index` in `index2`

    Examples
    --------
    >>> index = np.array([6, 0, 4, 2, 7, 1, 5, 3])
    >>> chunks = ((4, 4),)
    >>> a, b = make_block_sorted_slices(index, chunks)

    Notice that the first set of 4 items are sorted, and the
    second set of 4 items are sorted.

    >>> a
    array([0, 2, 4, 6, 1, 3, 5, 7])
    >>> b
    array([3, 0, 2, 1, 7, 4, 6, 5])
    """
    from dask.array.core import slices_from_chunks
    slices = slices_from_chunks(chunks)
    if len(slices[0]) > 1:
        slices = [slice_[0] for slice_ in slices]
    offsets = np.roll(np.cumsum(chunks[0]), 1)
    offsets[0] = 0
    index2 = np.empty_like(index)
    index3 = np.empty_like(index)
    for slice_, offset in zip(slices, offsets):
        a = index[slice_]
        b = np.sort(a)
        c = offset + np.argsort(b.take(np.argsort(a)))
        index2[slice_] = b
        index3[slice_] = c
    return (index2, index3)