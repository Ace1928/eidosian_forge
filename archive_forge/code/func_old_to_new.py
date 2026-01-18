from __future__ import annotations
import heapq
import math
from functools import reduce
from itertools import chain, count, product
from operator import add, itemgetter, mul
from warnings import warn
import numpy as np
import tlz as toolz
from tlz import accumulate
from dask import config
from dask.array.chunk import getitem
from dask.array.core import Array, concatenate3, normalize_chunks
from dask.array.utils import validate_axis
from dask.array.wrap import empty
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import parse_bytes
def old_to_new(old_chunks, new_chunks):
    """Helper to build old_chunks to new_chunks.

    Handles missing values, as long as the dimension with the missing chunk values
    is unchanged.

    Notes
    -----
    This function expects that the arguments have been pre-processed by
    :func:`dask.array.core.normalize_chunks`. In particular any ``nan`` values should
    have been replaced (and are so by :func:`dask.array.core.normalize_chunks`)
    by the canonical ``np.nan``. It also expects that the arguments have been validated
    with `_validate_rechunk` and rechunking is thus possible.

    Examples
    --------
    >>> old = ((10, 10, 10, 10, 10), )
    >>> new = ((25, 5, 20), )
    >>> old_to_new(old, new)  # doctest: +NORMALIZE_WHITESPACE
    [[[(0, slice(0, 10, None)), (1, slice(0, 10, None)), (2, slice(0, 5, None))],
      [(2, slice(5, 10, None))],
      [(3, slice(0, 10, None)), (4, slice(0, 10, None))]]]
    """

    def is_unknown(dim):
        return any((math.isnan(chunk) for chunk in dim))
    dims_unknown = [is_unknown(dim) for dim in old_chunks]
    known_indices = []
    unknown_indices = []
    for i, unknown in enumerate(dims_unknown):
        if unknown:
            unknown_indices.append(i)
        else:
            known_indices.append(i)
    old_known = [old_chunks[i] for i in known_indices]
    new_known = [new_chunks[i] for i in known_indices]
    cmos = cumdims_label(old_known, 'o')
    cmns = cumdims_label(new_known, 'n')
    sliced = [None] * len(old_chunks)
    for i, cmo, cmn in zip(known_indices, cmos, cmns):
        sliced[i] = _intersect_1d(_breakpoints(cmo, cmn))
    for i in unknown_indices:
        dim = old_chunks[i]
        extra = [[(j, slice(0, size if not math.isnan(size) else None))] for j, size in enumerate(dim)]
        sliced[i] = extra
    assert all((x is not None for x in sliced))
    return sliced