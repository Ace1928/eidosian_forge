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
@pytest.mark.parametrize('shape_chunks', [((50, 4), (10, 2)), ((50,), (10,))])
@pytest.mark.parametrize('dtype', ['u4', np.float32, None, np.int64])
def test_nan_empty_like(shape_chunks, dtype):
    shape, chunks = shape_chunks
    x1 = da.random.standard_normal(size=shape, chunks=chunks)
    y1 = x1[x1 < 0.5]
    x2 = x1.compute()
    y2 = x2[x2 < 0.5]
    a_da = da.empty_like(y1, dtype=dtype).compute()
    a_np = np.empty_like(y2, dtype=dtype)
    assert a_da.shape == a_np.shape
    assert a_da.dtype == a_np.dtype