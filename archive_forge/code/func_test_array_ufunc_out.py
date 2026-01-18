from __future__ import annotations
import pickle
import warnings
from functools import partial
from operator import add
import pytest
import dask.array as da
from dask.array.ufunc import da_frompyfunc
from dask.array.utils import assert_eq
from dask.base import tokenize
def test_array_ufunc_out():
    x = da.arange(10, chunks=(5,))
    np.sin(x, out=x)
    np.add(x, 10, out=x)
    assert_eq(x, np.sin(np.arange(10)) + 10)