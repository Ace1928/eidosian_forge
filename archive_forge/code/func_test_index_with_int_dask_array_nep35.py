from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.utils import assert_eq
@pytest.mark.parametrize('idx_chunks', [None, 3, 2, 1])
@pytest.mark.parametrize('x_chunks', [(3, 5), (2, 3), (1, 2), (1, 1)])
def test_index_with_int_dask_array_nep35(x_chunks, idx_chunks):
    x = cupy.array([[10, 20, 30, 40, 50], [60, 70, 80, 90, 100], [110, 120, 130, 140, 150]])
    orig_idx = np.array([3, 0, 1])
    expect = cupy.array([[40, 10, 20], [90, 60, 70], [140, 110, 120]])
    if x_chunks is not None:
        x = da.from_array(x, chunks=x_chunks)
    if idx_chunks is not None:
        idx = da.from_array(orig_idx, chunks=idx_chunks)
    else:
        idx = orig_idx
    assert_eq(x[:, idx], expect)
    assert_eq(x.T[idx, :], expect.T)
    orig_idx = cupy.array(orig_idx)
    if idx_chunks is not None:
        idx = da.from_array(orig_idx, chunks=idx_chunks)
    else:
        idx = orig_idx
    assert_eq(x[:, idx], expect)
    assert_eq(x.T[idx, :], expect.T)