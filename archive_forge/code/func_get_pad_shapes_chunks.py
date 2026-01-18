from __future__ import annotations
import itertools
from collections.abc import Sequence
from functools import partial
from itertools import product
from numbers import Integral, Number
import numpy as np
from tlz import sliding_window
from dask.array import chunk
from dask.array.backends import array_creation_dispatch
from dask.array.core import (
from dask.array.numpy_compat import AxisError
from dask.array.ufunc import greater_equal, rint
from dask.array.utils import meta_from_array
from dask.array.wrap import empty, full, ones, zeros
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import cached_cumsum, derived_from, is_cupy_type
def get_pad_shapes_chunks(array, pad_width, axes):
    """
    Helper function for finding shapes and chunks of end pads.
    """
    pad_shapes = [list(array.shape), list(array.shape)]
    pad_chunks = [list(array.chunks), list(array.chunks)]
    for d in axes:
        for i in range(2):
            pad_shapes[i][d] = pad_width[d][i]
            pad_chunks[i][d] = (pad_width[d][i],)
    pad_shapes = [tuple(s) for s in pad_shapes]
    pad_chunks = [tuple(c) for c in pad_chunks]
    return (pad_shapes, pad_chunks)