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
def test_blockwise_no_args():

    def f():
        return np.ones((2, 3), np.float32)
    x = da.blockwise(f, 'ab', new_axes={'a': 2, 'b': (3, 3)}, dtype=np.float32)
    assert x.chunks == ((2,), (3, 3))
    assert_eq(x, np.ones((2, 6), np.float32))