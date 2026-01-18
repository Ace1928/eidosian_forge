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
def slice_slices_and_integers(out_name, in_name, blockdims, index):
    """
    Dask array indexing with slices and integers

    See Also
    --------

    _slice_1d
    """
    from dask.array.core import unknown_chunk_message
    shape = tuple((cached_cumsum(dim, initial_zero=True)[-1] for dim in blockdims))
    for dim, ind in zip(shape, index):
        if np.isnan(dim) and ind != slice(None, None, None):
            raise ValueError(f'Arrays chunk sizes are unknown: {shape}{unknown_chunk_message}')
    assert all((isinstance(ind, (slice, Integral)) for ind in index))
    assert len(index) == len(blockdims)
    block_slices = list(map(_slice_1d, shape, blockdims, index))
    sorted_block_slices = [sorted(i.items()) for i in block_slices]
    in_names = list(product([in_name], *[pluck(0, s) for s in sorted_block_slices]))
    out_names = list(product([out_name], *[range(len(d))[::-1] if i.step and i.step < 0 else range(len(d)) for d, i in zip(block_slices, index) if not isinstance(i, Integral)]))
    all_slices = list(product(*[pluck(1, s) for s in sorted_block_slices]))
    dsk_out = {out_name: (getitem, in_name, slices) for out_name, in_name, slices in zip(out_names, in_names, all_slices)}
    new_blockdims = [new_blockdim(d, db, i) for d, i, db in zip(shape, index, blockdims) if not isinstance(i, Integral)]
    return (dsk_out, new_blockdims)