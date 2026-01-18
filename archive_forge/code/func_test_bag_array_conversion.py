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
def test_bag_array_conversion():
    import dask.bag as db
    b = db.range(10, npartitions=1)
    x, = b.map_partitions(np.asarray).to_delayed()
    x, = (da.from_delayed(a, shape=(10,), dtype=int) for a in [x])
    z = da.concatenate([x])
    assert_eq(z, np.arange(10), check_graph=False)