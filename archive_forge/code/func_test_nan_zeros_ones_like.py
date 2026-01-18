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
@pytest.mark.parametrize('fn', ['zeros_like', 'ones_like'])
@pytest.mark.parametrize('shape_chunks', [((50, 4), (10, 2)), ((50,), (10,))])
@pytest.mark.parametrize('dtype', ['u4', np.float32, None, np.int64])
def test_nan_zeros_ones_like(fn, shape_chunks, dtype):
    dafn = getattr(da, fn)
    npfn = getattr(np, fn)
    shape, chunks = shape_chunks
    x1 = da.random.standard_normal(size=shape, chunks=chunks)
    y1 = x1[x1 < 0.5]
    x2 = x1.compute()
    y2 = x2[x2 < 0.5]
    assert_eq(dafn(y1, dtype=dtype), npfn(y2, dtype=dtype))