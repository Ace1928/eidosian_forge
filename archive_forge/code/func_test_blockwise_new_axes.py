from __future__ import annotations
import collections
from operator import add
import numpy as np
import pytest
import dask
import dask.array as da
from dask.array.utils import assert_eq
from dask.blockwise import (
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import dec, hlg_layer_topological, inc
def test_blockwise_new_axes():

    def f(x):
        return x[:, None] * np.ones((1, 7))
    x = da.ones(5, chunks=2)
    y = da.blockwise(f, 'aq', x, 'a', new_axes={'q': 7}, concatenate=True, dtype=x.dtype)
    assert y.chunks == ((2, 2, 1), (7,))
    assert_eq(y, np.ones((5, 7)))

    def f(x):
        return x[None, :] * np.ones((7, 1))
    x = da.ones(5, chunks=2)
    y = da.blockwise(f, 'qa', x, 'a', new_axes={'q': 7}, concatenate=True, dtype=x.dtype)
    assert y.chunks == ((7,), (2, 2, 1))
    assert_eq(y, np.ones((7, 5)))

    def f(x):
        y = x.sum(axis=1)
        return y[:, None] * np.ones((1, 5))
    x = da.ones((4, 6), chunks=(2, 2))
    y = da.blockwise(f, 'aq', x, 'ab', new_axes={'q': 5}, concatenate=True, dtype=x.dtype)
    assert y.chunks == ((2, 2), (5,))
    assert_eq(y, np.ones((4, 5)) * 6)