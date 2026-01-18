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
@pytest.mark.parametrize('dtype', [None, 'f8', 'i8'])
@pytest.mark.parametrize('func, kwargs', [(lambda x, y: x + y, {}), (lambda x, y, c=1: x + c * y, {}), (lambda x, y, c=1: x + c * y, {'c': 3})])
def test_fromfunction(func, dtype, kwargs):
    a = np.fromfunction(func, shape=(5, 5), dtype=dtype, **kwargs)
    d = da.fromfunction(func, shape=(5, 5), chunks=(2, 2), dtype=dtype, **kwargs)
    assert_eq(d, a)
    d2 = da.fromfunction(func, shape=(5, 5), chunks=(2, 2), dtype=dtype, **kwargs)
    assert same_keys(d, d2)