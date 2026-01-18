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
def find_merge_rechunk(old_chunks, new_chunks, block_size_limit):
    """
    Find an intermediate rechunk that would merge some adjacent blocks
    together in order to get us nearer the *new_chunks* target, without
    violating the *block_size_limit* (in number of elements).
    """
    ndim = len(old_chunks)
    old_largest_width = [max(c) for c in old_chunks]
    new_largest_width = [max(c) for c in new_chunks]
    graph_size_effect = {dim: len(nc) / len(oc) for dim, (oc, nc) in enumerate(zip(old_chunks, new_chunks))}
    block_size_effect = {dim: new_largest_width[dim] / (old_largest_width[dim] or 1) for dim in range(ndim)}
    merge_candidates = [dim for dim in range(ndim) if graph_size_effect[dim] <= 1.0]

    def key(k):
        gse = graph_size_effect[k]
        bse = block_size_effect[k]
        if bse == 1:
            bse = 1 + 1e-09
        return np.log(gse) / np.log(bse) if bse > 0 else 0
    sorted_candidates = sorted(merge_candidates, key=key)
    largest_block_size = reduce(mul, old_largest_width)
    chunks = list(old_chunks)
    memory_limit_hit = False
    for dim in sorted_candidates:
        new_largest_block_size = largest_block_size * new_largest_width[dim] // (old_largest_width[dim] or 1)
        if new_largest_block_size <= block_size_limit:
            chunks[dim] = new_chunks[dim]
            largest_block_size = new_largest_block_size
        else:
            largest_width = old_largest_width[dim]
            chunk_limit = int(block_size_limit * largest_width / largest_block_size)
            c = divide_to_width(new_chunks[dim], chunk_limit)
            if len(c) <= len(old_chunks[dim]):
                chunks[dim] = c
                largest_block_size = largest_block_size * max(c) // largest_width
            memory_limit_hit = True
    assert largest_block_size == _largest_block_size(chunks)
    assert largest_block_size <= block_size_limit
    return (tuple(chunks), memory_limit_hit)