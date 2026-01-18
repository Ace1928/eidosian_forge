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
def arg_chunk(func, argfunc, x, axis, offset_info):
    arg_axis = None if len(axis) == x.ndim or x.ndim == 1 else axis[0]
    vals = func(x, axis=arg_axis, keepdims=True)
    arg = argfunc(x, axis=arg_axis, keepdims=True)
    if x.ndim > 0:
        if arg_axis is None:
            offset, total_shape = offset_info
            ind = np.unravel_index(arg.ravel()[0], x.shape)
            total_ind = tuple((o + i for o, i in zip(offset, ind)))
            arg[:] = np.ravel_multi_index(total_ind, total_shape)
        else:
            arg += offset_info
    if isinstance(vals, np.ma.masked_array):
        if 'min' in argfunc.__name__:
            fill_value = np.ma.minimum_fill_value(vals)
        else:
            fill_value = np.ma.maximum_fill_value(vals)
        vals = np.ma.filled(vals, fill_value)
    try:
        result = np.empty_like(vals, shape=vals.shape, dtype=[('vals', vals.dtype), ('arg', arg.dtype)])
    except TypeError:
        result = dict()
    result['vals'] = vals
    result['arg'] = arg
    return result