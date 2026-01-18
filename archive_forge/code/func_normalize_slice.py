from __future__ import annotations
from collections.abc import Callable
from itertools import zip_longest
from numbers import Integral
from typing import Any
import numpy as np
from dask import config
from dask.array.chunk import getitem
from dask.array.core import getter, getter_inline, getter_nofancy
from dask.blockwise import fuse_roots, optimize_blockwise
from dask.core import flatten, reverse_dict
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable, fuse, inline_functions
from dask.utils import ensure_dict
def normalize_slice(s):
    """Replace Nones in slices with integers

    >>> normalize_slice(slice(None, None, None))
    slice(0, None, 1)
    """
    start, stop, step = (s.start, s.stop, s.step)
    if start is None:
        start = 0
    if step is None:
        step = 1
    if start < 0 or step < 0 or (stop is not None and stop < 0):
        raise NotImplementedError()
    return slice(start, stop, step)