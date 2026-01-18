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
def test_fuse_slices_with_alias(getter, getitem):
    dsk = {'x': np.arange(16).reshape((4, 4)), ('dx', 0, 0): (getter, 'x', (slice(0, 4), slice(0, 4))), ('alias', 0, 0): ('dx', 0, 0), ('dx2', 0): (getitem, ('alias', 0, 0), (slice(None), 0))}
    keys = [('dx2', 0)]
    dsk2 = optimize(dsk, keys)
    assert len(dsk2) == 3
    fused_key = (dsk2.keys() - {'x', ('dx2', 0)}).pop()
    assert _check_get_task_eq(dsk2[fused_key], (getter, 'x', (slice(0, 4), 0)))