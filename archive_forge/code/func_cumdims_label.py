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
def cumdims_label(chunks, const):
    """Internal utility for cumulative sum with label.

    >>> cumdims_label(((5, 3, 3), (2, 2, 1)), 'n')  # doctest: +NORMALIZE_WHITESPACE
    [(('n', 0), ('n', 5), ('n', 8), ('n', 11)),
     (('n', 0), ('n', 2), ('n', 4), ('n', 5))]
    """
    return [tuple(zip((const,) * (1 + len(bds)), accumulate(add, (0,) + bds))) for bds in chunks]