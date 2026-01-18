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
def test_disable_lowlevel_fusion():
    """Check that by disabling fusion, the HLG survives through optimizations"""
    with dask.config.set({'optimization.fuse.active': False}):
        y = da.ones(3, chunks=(3,), dtype='int')
        optimize = y.__dask_optimize__
        dsk1 = y.__dask_graph__()
        dsk2 = optimize(dsk1, y.__dask_keys__())
        assert isinstance(dsk1, HighLevelGraph)
        assert isinstance(dsk2, HighLevelGraph)
        assert dsk1 == dsk2
        y = y.persist()
        assert isinstance(y.__dask_graph__(), HighLevelGraph)
        assert_eq(y, [1] * 3)