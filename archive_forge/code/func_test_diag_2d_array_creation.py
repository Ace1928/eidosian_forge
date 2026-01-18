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
def test_diag_2d_array_creation(k):
    v = np.arange(11)
    assert_eq(da.diag(v, k), np.diag(v, k))
    v = da.arange(11, chunks=3)
    darr = da.diag(v, k)
    nparr = np.diag(v, k)
    assert_eq(darr, nparr)
    assert sorted(da.diag(v, k).dask) == sorted(da.diag(v, k).dask)
    v = v + v + 3
    darr = da.diag(v, k)
    nparr = np.diag(v, k)
    assert_eq(darr, nparr)
    v = da.arange(11, chunks=11)
    darr = da.diag(v, k)
    nparr = np.diag(v, k)
    assert_eq(darr, nparr)
    assert sorted(da.diag(v, k).dask) == sorted(da.diag(v, k).dask)