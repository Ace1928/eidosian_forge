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
def test_fft2n_shapes(funcname):
    da_fft = getattr(dask.array.fft, funcname)
    np_fft = getattr(np.fft, funcname)
    assert_eq(da_fft(darr3), np_fft(nparr))
    assert_eq(da_fft(darr3, (8, 9), axes=(1, 0)), np_fft(nparr, (8, 9), axes=(1, 0)))
    assert_eq(da_fft(darr3, (12, 11), axes=(1, 0)), np_fft(nparr, (12, 11), axes=(1, 0)))
    if NUMPY_GE_200 and funcname.endswith('fftn'):
        ctx = pytest.warns(DeprecationWarning, match='`axes` should not be `None` if `s` is not `None`')
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        expect = np_fft(nparr, (8, 9))
    with ctx:
        actual = da_fft(darr3, (8, 9))
    assert_eq(expect, actual)