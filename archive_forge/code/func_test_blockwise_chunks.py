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
def test_blockwise_chunks():
    x = da.ones((5, 5), chunks=((2, 1, 2), (3, 2)))

    def double(a, axis=0):
        return np.concatenate([a, a], axis=axis)
    y = da.blockwise(double, 'ij', x, 'ij', adjust_chunks={'i': lambda n: 2 * n}, axis=0, dtype=x.dtype)
    assert y.chunks == ((4, 2, 4), (3, 2))
    assert_eq(y, np.ones((10, 5)))
    y = da.blockwise(double, 'ij', x, 'ij', adjust_chunks={'j': lambda n: 2 * n}, axis=1, dtype=x.dtype)
    assert y.chunks == ((2, 1, 2), (6, 4))
    assert_eq(y, np.ones((5, 10)))
    x = da.ones((10, 10), chunks=(5, 5))
    y = da.blockwise(double, 'ij', x, 'ij', axis=0, adjust_chunks={'i': 10}, dtype=x.dtype)
    assert y.chunks == ((10, 10), (5, 5))
    assert_eq(y, np.ones((20, 10)))
    y = da.blockwise(double, 'ij', x, 'ij', axis=0, adjust_chunks={'i': (10, 10)}, dtype=x.dtype)
    assert y.chunks == ((10, 10), (5, 5))
    assert_eq(y, np.ones((20, 10)))