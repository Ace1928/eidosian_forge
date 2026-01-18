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
def plan_rechunk(old_chunks, new_chunks, itemsize, threshold=None, block_size_limit=None):
    """Plan an iterative rechunking from *old_chunks* to *new_chunks*.
    The plan aims to minimize the rechunk graph size.

    Parameters
    ----------
    itemsize: int
        The item size of the array
    threshold: int
        The graph growth factor under which we don't bother
        introducing an intermediate step
    block_size_limit: int
        The maximum block size (in bytes) we want to produce during an
        intermediate step

    Notes
    -----
    No intermediate steps will be planned if any dimension of ``old_chunks``
    is unknown.
    """
    threshold = threshold or config.get('array.rechunk.threshold')
    block_size_limit = block_size_limit or config.get('array.chunk-size')
    if isinstance(block_size_limit, str):
        block_size_limit = parse_bytes(block_size_limit)
    has_nans = (any((math.isnan(y) for y in x)) for x in old_chunks)
    if len(new_chunks) <= 1 or not all(new_chunks) or any(has_nans):
        return [new_chunks]
    block_size_limit /= itemsize
    largest_old_block = _largest_block_size(old_chunks)
    largest_new_block = _largest_block_size(new_chunks)
    block_size_limit = max([block_size_limit, largest_old_block, largest_new_block])
    graph_size_threshold = threshold * (_number_of_blocks(old_chunks) + _number_of_blocks(new_chunks))
    current_chunks = old_chunks
    first_pass = True
    steps = []
    while True:
        graph_size = estimate_graph_size(current_chunks, new_chunks)
        if graph_size < graph_size_threshold:
            break
        if first_pass:
            chunks = current_chunks
        else:
            chunks = find_split_rechunk(current_chunks, new_chunks, graph_size * threshold)
        chunks, memory_limit_hit = find_merge_rechunk(chunks, new_chunks, block_size_limit)
        if chunks == current_chunks and (not first_pass) or chunks == new_chunks:
            break
        if chunks != current_chunks:
            steps.append(chunks)
        current_chunks = chunks
        if not memory_limit_hit:
            break
        first_pass = False
    return steps + [new_chunks]