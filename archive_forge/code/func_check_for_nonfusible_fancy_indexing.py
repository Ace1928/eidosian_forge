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
def check_for_nonfusible_fancy_indexing(fancy, normal):
    for f, n in zip_longest(fancy, normal, fillvalue=slice(None)):
        if type(f) is not list and isinstance(n, Integral):
            raise NotImplementedError("Can't handle normal indexing with integers and fancy indexing if the integers and fancy indices don't align with the same dimensions.")