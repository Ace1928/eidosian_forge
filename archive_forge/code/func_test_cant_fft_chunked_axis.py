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
def test_cant_fft_chunked_axis(funcname):
    da_fft = getattr(da.fft, funcname)
    bad_darr = da.from_array(nparr, chunks=(5, 5))
    for i in range(bad_darr.ndim):
        with pytest.raises(ValueError):
            da_fft(bad_darr, axis=i)