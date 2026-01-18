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
@pytest.mark.parametrize('funcname1, funcname2', [('fftshift', 'ifftshift'), ('ifftshift', 'fftshift')])
@pytest.mark.parametrize('axes', [None, 0, 1, 2, (0, 1), (1, 2), (0, 2), (0, 1, 2)])
@pytest.mark.parametrize('shape, chunks', [[(5, 6, 7), (2, 3, 4)], [(5, 6, 7), (2, 6, 4)], [(5, 6, 7), (5, 6, 7)]])
def test_fftshift_identity(funcname1, funcname2, shape, chunks, axes):
    da_func1 = getattr(da.fft, funcname1)
    da_func2 = getattr(da.fft, funcname2)
    a = np.arange(np.prod(shape)).reshape(shape)
    d = da.from_array(a, chunks=chunks)
    d_r = da_func1(da_func2(d, axes), axes)
    for each_d_chunks, each_d_r_chunks in zip(d.chunks, d_r.chunks):
        if len(each_d_chunks) == 1:
            assert len(each_d_r_chunks) == 1
            assert each_d_r_chunks == each_d_chunks
        else:
            assert len(each_d_r_chunks) != 1
    assert_eq(d_r, d)