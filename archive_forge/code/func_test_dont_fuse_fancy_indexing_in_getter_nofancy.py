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
def test_dont_fuse_fancy_indexing_in_getter_nofancy(getitem, getter_nofancy):
    dsk = {'a': (getitem, (getter_nofancy, 'x', (slice(10, 20, None), slice(100, 200, None))), ([1, 3], slice(50, 60, None)))}
    _assert_getter_dsk_eq(optimize_slices(dsk), dsk)
    dsk = {'a': (getitem, (getter_nofancy, 'x', [1, 2, 3]), 0)}
    _assert_getter_dsk_eq(optimize_slices(dsk), dsk)