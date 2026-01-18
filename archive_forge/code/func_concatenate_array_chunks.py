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
def concatenate_array_chunks(x):
    """Concatenate the multidimensional chunks of an array.

    Can be used on chunks with unknown sizes.

    Parameters
    ----------
    x : dask array

    Returns
    -------
    dask array
        The concatenated dask array with one chunk.

    """
    from dask.array.core import Array, concatenate3
    if x.npartitions == 1:
        return x
    name = 'concatenate3-' + tokenize(x)
    d = {(name, 0): (concatenate3, x.__dask_keys__())}
    graph = HighLevelGraph.from_collections(name, d, dependencies=[x])
    chunks = x.shape
    if not chunks:
        chunks = (1,)
    return Array(graph, name, chunks=(chunks,), dtype=x.dtype)