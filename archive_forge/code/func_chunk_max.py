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
def chunk_max(x, axis=None, keepdims=None):
    """Version of np.max which ignores size 0 arrays"""
    if x.size == 0:
        return array_safe([], x, ndmin=x.ndim, dtype=x.dtype)
    else:
        return np.max(x, axis=axis, keepdims=keepdims)