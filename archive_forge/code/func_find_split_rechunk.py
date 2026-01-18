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
def find_split_rechunk(old_chunks, new_chunks, graph_size_limit):
    """
    Find an intermediate rechunk that would split some chunks to
    get us nearer *new_chunks*, without violating the *graph_size_limit*.
    """
    ndim = len(old_chunks)
    chunks = list(old_chunks)
    for dim in range(ndim):
        graph_size = estimate_graph_size(chunks, new_chunks)
        if graph_size > graph_size_limit:
            break
        if len(old_chunks[dim]) > len(new_chunks[dim]):
            continue
        max_number = int(len(old_chunks[dim]) * graph_size_limit / graph_size)
        c = merge_to_number(new_chunks[dim], max_number)
        assert len(c) <= max_number
        if len(c) >= len(old_chunks[dim]) and max(c) <= max(old_chunks[dim]):
            chunks[dim] = c
    return tuple(chunks)