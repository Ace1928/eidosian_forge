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
def slice_with_newaxes(out_name, in_name, blockdims, index, itemsize):
    """
    Handle indexing with Nones

    Strips out Nones then hands off to slice_wrap_lists
    """
    index2 = tuple((ind for ind in index if ind is not None))
    where_none = [i for i, ind in enumerate(index) if ind is None]
    where_none_orig = list(where_none)
    for i, x in enumerate(where_none):
        n = sum((isinstance(ind, Integral) for ind in index[:x]))
        if n:
            where_none[i] -= n
    dsk, blockdims2 = slice_wrap_lists(out_name, in_name, blockdims, index2, itemsize)
    if where_none:
        expand = expander(where_none)
        expand_orig = expander(where_none_orig)
        dsk2 = {(out_name,) + expand(k[1:], 0): v[:2] + (expand_orig(v[2], None),) for k, v in dsk.items() if k[0] == out_name}
        dsk3 = merge(dsk2, {k: v for k, v in dsk.items() if k[0] != out_name})
        blockdims3 = expand(blockdims2, (1,))
        return (dsk3, blockdims3)
    else:
        return (dsk, blockdims2)