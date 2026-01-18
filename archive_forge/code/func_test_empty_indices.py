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
def test_empty_indices():
    darr = da.indices(tuple(), chunks=tuple())
    nparr = np.indices(tuple())
    assert darr.shape == nparr.shape
    assert darr.dtype == nparr.dtype
    assert_eq(darr, nparr)
    darr = da.indices(tuple(), float, chunks=tuple())
    nparr = np.indices(tuple(), float)
    assert darr.shape == nparr.shape
    assert darr.dtype == nparr.dtype
    assert_eq(darr, nparr)
    darr = da.indices((0,), float, chunks=(1,))
    nparr = np.indices((0,), float)
    assert darr.shape == nparr.shape
    assert darr.dtype == nparr.dtype
    assert_eq(darr, nparr)
    darr = da.indices((0, 1, 2), float, chunks=(1, 1, 2))
    nparr = np.indices((0, 1, 2), float)
    assert darr.shape == nparr.shape
    assert darr.dtype == nparr.dtype
    assert_eq(darr, nparr)