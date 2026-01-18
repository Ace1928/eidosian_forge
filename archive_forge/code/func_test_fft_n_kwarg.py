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
@pytest.mark.parametrize('funcname', all_1d_funcnames)
def test_fft_n_kwarg(funcname):
    da_fft = getattr(da.fft, funcname)
    np_fft = getattr(np.fft, funcname)
    assert_eq(da_fft(darr, 5), np_fft(nparr, 5))
    assert_eq(da_fft(darr, 13), np_fft(nparr, 13))
    assert_eq(da_fft(darr2, axis=0), np_fft(nparr, axis=0))
    assert_eq(da_fft(darr2, 5, axis=0), np_fft(nparr, 5, axis=0))
    assert_eq(da_fft(darr2, 13, axis=0), np_fft(nparr, 13, axis=0))
    assert_eq(da_fft(darr2, 12, axis=0), np_fft(nparr, 12, axis=0))