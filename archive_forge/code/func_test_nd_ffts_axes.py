from __future__ import annotations
import contextlib
from itertools import combinations_with_replacement
import numpy as np
import pytest
import dask.array as da
import dask.array.fft
from dask.array.core import normalize_chunks
from dask.array.fft import fft_wrap
from dask.array.numpy_compat import NUMPY_GE_200
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('funcname', all_nd_funcnames)
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_nd_ffts_axes(funcname, dtype):
    np_fft = getattr(np.fft, funcname)
    da_fft = getattr(da.fft, funcname)
    shape = (7, 8, 9)
    chunk_size = (3, 3, 3)
    a = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    d = da.from_array(a, chunks=chunk_size)
    for num_axes in range(1, d.ndim):
        for axes in combinations_with_replacement(range(d.ndim), num_axes):
            cs = list(chunk_size)
            for i in axes:
                cs[i] = shape[i]
            d2 = d.rechunk(cs)
            if len(set(axes)) < len(axes):
                with pytest.raises(ValueError):
                    da_fft(d2, axes=axes)
            else:
                r = da_fft(d2, axes=axes)
                er = np_fft(a, axes=axes)
                assert r.dtype == er.dtype
                assert r.shape == er.shape
                assert_eq(r, er)