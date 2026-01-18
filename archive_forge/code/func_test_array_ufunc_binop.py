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
def test_array_ufunc_binop():
    x = np.arange(25).reshape((5, 5))
    d = da.from_array(x, chunks=(2, 2))
    for func in [np.add, np.multiply]:
        assert isinstance(func(d, d), da.Array)
        assert_eq(func(d, d), func(x, x))
        assert isinstance(func.outer(d, d), da.Array)
        assert_eq(func.outer(d, d), func.outer(x, x))