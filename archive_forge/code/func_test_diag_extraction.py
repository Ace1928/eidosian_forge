from __future__ import annotations
import pytest
import numpy as np
import pytest
from tlz import concat
import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.array.numpy_compat import AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('k', [0, 3, -3, 8])
def test_diag_extraction(k):
    x = np.arange(64).reshape((8, 8))
    assert_eq(da.diag(x, k), np.diag(x, k))
    d = da.from_array(x, chunks=(4, 4))
    assert_eq(da.diag(d, k), np.diag(x, k))
    d = da.from_array(x, chunks=((3, 2, 3), (4, 1, 2, 1)))
    assert_eq(da.diag(d, k), np.diag(x, k))
    y = np.arange(5 * 8).reshape((5, 8))
    assert_eq(da.diag(y, k), np.diag(y, k))
    d = da.from_array(y, chunks=(4, 4))
    assert_eq(da.diag(d, k), np.diag(y, k))
    d = da.from_array(y, chunks=((3, 2), (4, 1, 2, 1)))
    assert_eq(da.diag(d, k), np.diag(y, k))