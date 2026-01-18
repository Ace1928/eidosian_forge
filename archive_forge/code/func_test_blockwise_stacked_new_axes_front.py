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
@pytest.mark.parametrize('concatenate', [True, False])
def test_blockwise_stacked_new_axes_front(concatenate):

    def f(x):
        if isinstance(x, list):
            x = np.concatenate(x)
        return x[None, ...] * np.ones(7)[(slice(None),) + (None,) * x.ndim]
    x = da.ones(5, chunks=2)
    y = da.blockwise(f, 'qa', x, 'a', new_axes={'q': 7}, concatenate=concatenate, dtype=x.dtype)
    z = da.blockwise(f, 'qab', y, 'ab', new_axes={'q': 7}, concatenate=concatenate, dtype=x.dtype)
    assert z.chunks == ((7,), (7,), (2, 2, 1))
    assert_eq(z, np.ones((7, 7, 5)))
    w = da.blockwise(lambda x: x[:, 0, 0], 'a', z, 'abc', dtype=x.dtype, concatenate=True)
    assert w.chunks == ((7,),)
    assert_eq(w, np.ones((7,)))