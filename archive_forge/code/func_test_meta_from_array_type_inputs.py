from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq, meta_from_array
from dask.local import get_sync
def test_meta_from_array_type_inputs():
    x = meta_from_array(np.ndarray, ndim=2, dtype=np.float32)
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2
    assert x.dtype == np.float32
    x = da.Array({('x', 0, 0): (np.ones, (5, 5))}, name='x', chunks=(5, 5), shape=(5, 5), meta=np.ndarray, dtype=float)
    assert_eq(x, x)
    assert da.from_array(np.ones(5).astype(np.int32), meta=np.ndarray).dtype == np.int32