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
def test_fuse_roots_annotations():
    x = da.ones(10, chunks=(2,))
    y = da.zeros(10, chunks=(2,))
    with dask.annotate(foo='bar'):
        y = y ** 2
    z = x + 1 + 2 * y
    hlg = dask.blockwise.optimize_blockwise(z.dask)
    assert len(hlg.layers) == 3
    assert {'foo': 'bar'} in [l.annotations for l in hlg.layers.values()]
    za = da.Array(hlg, z.name, z.chunks, z.dtype)
    assert_eq(za, z)