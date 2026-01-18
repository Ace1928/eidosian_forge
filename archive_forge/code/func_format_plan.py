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
def format_plan(plan):
    """
    >>> format_plan([((10, 10, 10), (15, 15)), ((30,), (10, 10, 10))])
    [(3*[10], 2*[15]), ([30], 3*[10])]
    """
    return [format_chunks(c) for c in plan]