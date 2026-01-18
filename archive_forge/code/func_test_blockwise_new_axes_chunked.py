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
def test_blockwise_new_axes_chunked():

    def f(x):
        return x[None, :] * 2
    x = da.arange(0, 6, 1, chunks=2, dtype=np.int32)
    y = da.blockwise(f, 'qa', x, 'a', new_axes={'q': (1, 1)}, dtype=x.dtype)
    assert y.chunks == ((1, 1), (2, 2, 2))
    assert_eq(y, np.array([[0, 2, 4, 6, 8, 10], [0, 2, 4, 6, 8, 10]], np.int32))