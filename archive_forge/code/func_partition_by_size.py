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
def partition_by_size(sizes, seq):
    """

    >>> partition_by_size([10, 20, 10], [1, 5, 9, 12, 29, 35])
    [array([1, 5, 9]), array([ 2, 19]), array([5])]
    """
    if not is_arraylike(seq):
        seq = np.asanyarray(seq)
    left = np.empty(len(sizes) + 1, dtype=int)
    left[0] = 0
    right = np.cumsum(sizes, out=left[1:])
    locations = np.empty(len(sizes) + 1, dtype=int)
    locations[0] = 0
    locations[1:] = np.searchsorted(seq, right)
    return [seq[j:k] - l for j, k, l in zip(locations[:-1], locations[1:], left)]