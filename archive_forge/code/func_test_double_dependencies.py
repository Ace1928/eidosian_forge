from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.chunk import getitem as da_getitem
from dask.array.core import getter as da_getter
from dask.array.core import getter_nofancy as da_getter_nofancy
from dask.array.optimization import (
from dask.array.utils import assert_eq
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable, fuse
from dask.utils import SerializableLock
def test_double_dependencies():
    x = np.arange(56).reshape((7, 8))
    d = da.from_array(x, chunks=(4, 4))
    X = d + 1
    X = da.dot(X, X.T)
    assert_eq(X.compute(optimize_graph=False), X)