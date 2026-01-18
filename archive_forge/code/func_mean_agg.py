from __future__ import annotations
import builtins
import contextlib
import math
import operator
import warnings
from collections.abc import Iterable
from functools import partial
from itertools import product, repeat
from numbers import Integral, Number
import numpy as np
from tlz import accumulate, compose, drop, get, partition_all, pluck
from dask import config
from dask.array import chunk
from dask.array.blockwise import blockwise
from dask.array.core import (
from dask.array.creation import arange, diagonal
from dask.array.dispatch import divide_lookup, nannumel_lookup, numel_lookup
from dask.array.numpy_compat import ComplexWarning
from dask.array.utils import (
from dask.array.wrap import ones, zeros
from dask.base import tokenize
from dask.blockwise import lol_tuples
from dask.highlevelgraph import HighLevelGraph
from dask.utils import (
def mean_agg(pairs, dtype='f8', axis=None, computing_meta=False, **kwargs):
    ns = deepmap(lambda pair: pair['n'], pairs) if not computing_meta else pairs
    n = _concatenate2(ns, axes=axis)
    n = np.sum(n, axis=axis, dtype=dtype, **kwargs)
    if computing_meta:
        return n
    totals = deepmap(lambda pair: pair['total'], pairs)
    total = _concatenate2(totals, axes=axis).sum(axis=axis, dtype=dtype, **kwargs)
    with np.errstate(divide='ignore', invalid='ignore'):
        return divide(total, n, dtype=dtype)