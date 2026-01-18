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
def test_inner_compute():
    x = da.ones(10, chunks=(5,)) + 1 + 2 + 3
    a = x.sum()
    y = x * 2 * 3 * 4
    b = y.sum()
    z = x * 2 * 3
    dask.compute(x, a, y, b, z)