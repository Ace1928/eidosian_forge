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
@pytest.fixture(params=[True, False])
def getter_nofancy(request):
    """
    Parameterized fixture for dask.array.chunk.getter_nofancy both alone (False)
    and wrapped in a SubgraphCallable (True).
    """
    yield _wrap_getter(da_getter_nofancy, request.param)