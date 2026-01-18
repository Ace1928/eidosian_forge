from __future__ import annotations
import random
import warnings
from bisect import bisect_left
from itertools import cycle
from operator import add, itemgetter
from tlz import accumulate, groupby, pluck, unique
from dask.core import istask
from dask.utils import apply, funcname, import_required
def fix_bounds(start, end, min_span):
    """Adjust end point to ensure span of at least `min_span`"""
    return (start, max(end, start + min_span))