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
def test_fuse_slice_with_lists():
    assert fuse_slice(slice(10, 20, 2), [1, 2, 3]) == [12, 14, 16]
    assert fuse_slice([10, 20, 30, 40, 50], [3, 1, 2]) == [40, 20, 30]
    assert fuse_slice([10, 20, 30, 40, 50], 3) == 40
    assert fuse_slice([10, 20, 30, 40, 50], -1) == 50
    assert fuse_slice([10, 20, 30, 40, 50], slice(1, None, 2)) == [20, 40]
    assert fuse_slice((slice(None), slice(0, 10), [1, 2, 3]), (slice(None), slice(1, 5), slice(None))) == (slice(0, None), slice(1, 5), [1, 2, 3])
    assert fuse_slice((slice(None), slice(None), [1, 2, 3]), (slice(None), slice(1, 5), 1)) == (slice(0, None), slice(1, 5), 2)