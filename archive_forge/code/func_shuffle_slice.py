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
def shuffle_slice(x, index):
    """A relatively efficient way to shuffle `x` according to `index`.

    Parameters
    ----------
    x : Array
    index : ndarray
        This should be an ndarray the same length as `x` containing
        each index position in ``range(0, len(x))``.

    Returns
    -------
    Array
    """
    from dask.array.core import PerformanceWarning
    chunks1 = chunks2 = x.chunks
    if x.ndim > 1:
        chunks1 = (chunks1[0],)
    index2, index3 = make_block_sorted_slices(index, chunks1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PerformanceWarning)
        return x[index2].rechunk(chunks2)[index3]