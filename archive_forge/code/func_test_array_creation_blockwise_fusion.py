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
def test_array_creation_blockwise_fusion():
    """
    Check that certain array creation routines work with blockwise and can be
    fused with other blockwise operations.
    """
    x = da.ones(3, chunks=(3,))
    y = da.zeros(3, chunks=(3,))
    z = da.full(3, fill_value=2, chunks=(3,))
    a = x + y + z
    dsk1 = a.__dask_graph__()
    assert len(dsk1) == 5
    dsk2 = optimize_blockwise(dsk1)
    assert len(dsk2) == 1
    assert_eq(a, np.full(3, 3.0))