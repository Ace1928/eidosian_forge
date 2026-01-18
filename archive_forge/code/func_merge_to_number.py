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
def merge_to_number(desired_chunks, max_number):
    """Minimally merge the given chunks so as to drop the number of
    chunks below *max_number*, while minimizing the largest width.
    """
    if len(desired_chunks) <= max_number:
        return desired_chunks
    distinct = set(desired_chunks)
    if len(distinct) == 1:
        w = distinct.pop()
        n = len(desired_chunks)
        total = n * w
        desired_width = total // max_number
        width = w * (desired_width // w)
        adjust = (total - max_number * width) // w
        return (width + w,) * adjust + (width,) * (max_number - adjust)
    desired_width = sum(desired_chunks) // max_number
    nmerges = len(desired_chunks) - max_number
    heap = [(desired_chunks[i] + desired_chunks[i + 1], i, i + 1) for i in range(len(desired_chunks) - 1)]
    heapq.heapify(heap)
    chunks = list(desired_chunks)
    while nmerges > 0:
        width, i, j = heapq.heappop(heap)
        if chunks[j] == 0:
            j += 1
            while chunks[j] == 0:
                j += 1
            heapq.heappush(heap, (chunks[i] + chunks[j], i, j))
            continue
        elif chunks[i] + chunks[j] != width:
            heapq.heappush(heap, (chunks[i] + chunks[j], i, j))
            continue
        assert chunks[i] != 0
        chunks[i] = 0
        chunks[j] = width
        nmerges -= 1
    return tuple(filter(None, chunks))