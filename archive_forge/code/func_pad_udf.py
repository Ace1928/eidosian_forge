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
def pad_udf(array, pad_width, mode, **kwargs):
    """
    Helper function for padding boundaries with a user defined function.

    In cases where the padding requires a custom user defined function be
    applied to the array, this function assists in the prepping and
    application of this function to the Dask Array to construct the desired
    boundaries.
    """
    result = pad_edge(array, pad_width, 'constant', constant_values=0)
    chunks = result.chunks
    for d in range(result.ndim):
        result = result.rechunk(chunks[:d] + (result.shape[d:d + 1],) + chunks[d + 1:])
        result = result.map_blocks(wrapped_pad_func, name='pad', dtype=result.dtype, pad_func=mode, iaxis_pad_width=pad_width[d], iaxis=d, pad_func_kwargs=kwargs)
        result = result.rechunk(chunks)
    return result