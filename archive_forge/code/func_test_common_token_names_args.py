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
@pytest.mark.parametrize('name', ['_', '_0', '_1', '.', '.0'])
def test_common_token_names_args(name):
    x = np.array(['a', 'bb', 'ccc'], dtype=object)
    d = da.from_array(x, chunks=2)
    result = da.blockwise(add, 'i', d, 'i', name, None, dtype=object)
    expected = x + name
    assert_eq(result, expected)