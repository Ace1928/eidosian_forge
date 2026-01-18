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
@pytest.mark.parametrize('start,stop,step,dtype', [(0, 1, 1, None), (1.5, 2, 1, None), (1, 2.5, 1, None), (1, 2, 0.5, None), (np.float32(1), np.float32(2), np.float32(1), None), (np.int32(1), np.int32(2), np.int32(1), None), (np.uint32(1), np.uint32(2), np.uint32(1), None), (np.uint64(1), np.uint64(2), np.uint64(1), None), (np.uint32(1), np.uint32(2), np.uint32(1), np.uint32), (np.uint64(1), np.uint64(2), np.uint64(1), np.uint64)])
def test_arange_dtypes(start, stop, step, dtype):
    a_np = np.arange(start, stop, step, dtype=dtype)
    a_da = da.arange(start, stop, step, dtype=dtype, chunks=-1)
    assert_eq(a_np, a_da)