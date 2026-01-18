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
@pytest.mark.parametrize('which', ['getter', 'getter_nofancy'])
def test_dont_remove_no_op_slices_for_getter_or_getter_nofancy(which, getitem, getter, getter_nofancy):
    if which == 'getter':
        get = getter
    else:
        get = getter_nofancy
    null = slice(0, None)
    opts = [((get, 'x', null, False, False), (get, 'x', null, False, False)), ((getitem, (get, 'x', null, False, False), null), (get, 'x', null, False, False)), ((getitem, (get, 'x', (null, null), False, False), ()), (get, 'x', (null, null), False, False))]
    for orig, final in opts:
        _assert_getter_dsk_eq(optimize_slices({'a': orig}), {'a': final})