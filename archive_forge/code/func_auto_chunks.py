from __future__ import annotations
import contextlib
import math
import operator
import os
import pickle
import re
import sys
import traceback
import uuid
import warnings
from bisect import bisect
from collections.abc import (
from functools import partial, reduce, wraps
from itertools import product, zip_longest
from numbers import Integral, Number
from operator import add, mul
from threading import Lock
from typing import Any, TypeVar, Union, cast
import numpy as np
from numpy.typing import ArrayLike
from tlz import accumulate, concat, first, frequencies, groupby, partition
from tlz.curried import pluck
from dask import compute, config, core
from dask.array import chunk
from dask.array.chunk import getitem
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.dispatch import (  # noqa: F401
from dask.array.numpy_compat import _Recurser
from dask.array.slicing import replace_ellipsis, setitem_array, slice_array
from dask.base import (
from dask.blockwise import blockwise as core_blockwise
from dask.blockwise import broadcast_dimensions
from dask.context import globalmethod
from dask.core import quote
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.layers import ArraySliceDep, reshapelist
from dask.sizeof import sizeof
from dask.typing import Graph, Key, NestedKeys
from dask.utils import (
from dask.widgets import get_template
from dask.array.optimization import fuse_slice, optimize
from dask.array.blockwise import blockwise
from dask.array.utils import compute_meta, meta_from_array
def auto_chunks(chunks, shape, limit, dtype, previous_chunks=None):
    """Determine automatic chunks

    This takes in a chunks value that contains ``"auto"`` values in certain
    dimensions and replaces those values with concrete dimension sizes that try
    to get chunks to be of a certain size in bytes, provided by the ``limit=``
    keyword.  If multiple dimensions are marked as ``"auto"`` then they will
    all respond to meet the desired byte limit, trying to respect the aspect
    ratio of their dimensions in ``previous_chunks=``, if given.

    Parameters
    ----------
    chunks: Tuple
        A tuple of either dimensions or tuples of explicit chunk dimensions
        Some entries should be "auto"
    shape: Tuple[int]
    limit: int, str
        The maximum allowable size of a chunk in bytes
    previous_chunks: Tuple[Tuple[int]]

    See also
    --------
    normalize_chunks: for full docstring and parameters
    """
    if previous_chunks is not None:
        previous_chunks = tuple((c if isinstance(c, tuple) else (c,) for c in previous_chunks))
    chunks = list(chunks)
    autos = {i for i, c in enumerate(chunks) if c == 'auto'}
    if not autos:
        return tuple(chunks)
    if limit is None:
        limit = config.get('array.chunk-size')
    if isinstance(limit, str):
        limit = parse_bytes(limit)
    if dtype is None:
        raise TypeError('dtype must be known for auto-chunking')
    if dtype.hasobject:
        raise NotImplementedError('Can not use auto rechunking with object dtype. We are unable to estimate the size in bytes of object data')
    for x in tuple(chunks) + tuple(shape):
        if isinstance(x, Number) and np.isnan(x) or (isinstance(x, tuple) and np.isnan(x).any()):
            raise ValueError('Can not perform automatic rechunking with unknown (nan) chunk sizes.%s' % unknown_chunk_message)
    limit = max(1, limit)
    largest_block = math.prod((cs if isinstance(cs, Number) else max(cs) for cs in chunks if cs != 'auto'))
    if previous_chunks:
        result = {a: np.median(previous_chunks[a]) for a in autos}
        ideal_shape = []
        for i, s in enumerate(shape):
            chunk_frequencies = frequencies(previous_chunks[i])
            mode, count = max(chunk_frequencies.items(), key=lambda kv: kv[1])
            if mode > 1 and count >= len(previous_chunks[i]) / 2:
                ideal_shape.append(mode)
            else:
                ideal_shape.append(s)
        multiplier = _compute_multiplier(limit, dtype, largest_block, result)
        last_multiplier = 0
        last_autos = set()
        while multiplier != last_multiplier or autos != last_autos:
            last_multiplier = multiplier
            last_autos = set(autos)
            for a in sorted(autos):
                if ideal_shape[a] == 0:
                    result[a] = 0
                    continue
                proposed = result[a] * multiplier ** (1 / len(autos))
                if proposed > shape[a]:
                    autos.remove(a)
                    largest_block *= shape[a]
                    chunks[a] = shape[a]
                    del result[a]
                else:
                    result[a] = round_to(proposed, ideal_shape[a])
            multiplier = _compute_multiplier(limit, dtype, largest_block, result)
        for k, v in result.items():
            chunks[k] = v
        return tuple(chunks)
    else:
        if dtype.itemsize == 0:
            raise ValueError('auto-chunking with dtype.itemsize == 0 is not supported, please pass in `chunks` explicitly')
        size = (limit / dtype.itemsize / largest_block) ** (1 / len(autos))
        small = [i for i in autos if shape[i] < size]
        if small:
            for i in small:
                chunks[i] = (shape[i],)
            return auto_chunks(chunks, shape, limit, dtype)
        for i in autos:
            chunks[i] = round_to(size, shape[i])
        return tuple(chunks)