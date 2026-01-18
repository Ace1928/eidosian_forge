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
def pad_reuse(array, pad_width, mode, **kwargs):
    """
    Helper function for padding boundaries with values in the array.

    Handles the cases where the padding is constructed from values in
    the array. Namely by reflecting them or tiling them to create periodic
    boundary constraints.
    """
    if mode in {'reflect', 'symmetric'}:
        reflect_type = kwargs.get('reflect', 'even')
        if reflect_type == 'odd':
            raise NotImplementedError('`pad` does not support `reflect_type` of `odd`.')
        if reflect_type != 'even':
            raise ValueError('unsupported value for reflect_type, must be one of (`even`, `odd`)')
    result = np.empty(array.ndim * (3,), dtype=object)
    for idx in np.ndindex(result.shape):
        select = []
        orient = []
        for i, s, pw in zip(idx, array.shape, pad_width):
            if mode == 'wrap':
                pw = pw[::-1]
            if i < 1:
                if mode == 'reflect':
                    select.append(slice(1, pw[0] + 1, None))
                else:
                    select.append(slice(None, pw[0], None))
            elif i > 1:
                if mode == 'reflect':
                    select.append(slice(s - pw[1] - 1, s - 1, None))
                else:
                    select.append(slice(s - pw[1], None, None))
            else:
                select.append(slice(None))
            if i != 1 and mode in ['reflect', 'symmetric']:
                orient.append(slice(None, None, -1))
            else:
                orient.append(slice(None))
        select = tuple(select)
        orient = tuple(orient)
        if mode == 'wrap':
            idx = tuple((2 - i for i in idx))
        result[idx] = array[select][orient]
    result = block(result.tolist())
    return result