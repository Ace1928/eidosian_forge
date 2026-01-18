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
@pytest.mark.parametrize('funcname, kwargs', [('empty_like', {}), ('ones_like', {}), ('zeros_like', {}), ('full_like', {'fill_value': 5})])
@pytest.mark.parametrize('shape, chunks, out_shape', [((10, 10), (4, 4), None), ((10, 10), (4, 4), (20, 3)), ((10, 10), 4, 20), ((10, 10, 10), (4, 2), (5, 5)), ((2, 3, 5, 7), None, (3, 5, 7)), ((2, 3, 5, 7), (2, 5, 3), (3, 5, 7)), ((2, 3, 5, 7), (2, 5, 3, 'auto', 3), (11,) + (2, 3, 5, 7)), ((2, 3, 5, 7), 'auto', (3, 5, 7))])
@pytest.mark.parametrize('dtype', ['i4'])
def test_arr_like_shape(funcname, kwargs, shape, dtype, chunks, out_shape):
    np_func = getattr(np, funcname)
    da_func = getattr(da, funcname)
    a = np.random.randint(0, 10, shape).astype(dtype)
    np_r = np_func(a, shape=out_shape, **kwargs)
    da_r = da_func(a, chunks=chunks, shape=out_shape, **kwargs)
    assert np_r.shape == da_r.shape
    assert np_r.dtype == da_r.dtype
    if 'empty' not in funcname:
        assert_eq(np_r, da_r)