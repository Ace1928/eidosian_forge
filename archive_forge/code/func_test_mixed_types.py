from __future__ import annotations
import warnings
import pytest
import numpy as np
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('arg1', [pd.Series(np.abs(np.random.randn(100))), pd.DataFrame({'A': np.random.randint(1, 100, size=20), 'B': np.random.randint(1, 100, size=20), 'C': np.abs(np.random.randn(20))})])
@pytest.mark.parametrize('arg2', [2, dd.from_pandas(pd.Series([0]), 1).sum()])
@pytest.mark.parametrize('ufunc', _UFUNCS_2ARG)
def test_mixed_types(ufunc, arg1, arg2):
    npfunc = getattr(np, ufunc)
    dafunc = getattr(da, ufunc)
    dask = dd.from_pandas(arg1, 3)
    pandas_type = arg1.__class__
    dask_type = dask.__class__
    assert isinstance(dafunc(dask, arg2), dask_type)
    assert_eq(dafunc(dask, arg2), npfunc(dask, arg2))
    assert isinstance(npfunc(dask, arg2), dask_type)
    assert_eq(npfunc(dask, arg2), npfunc(arg1, arg2))
    assert isinstance(dafunc(arg1, arg2), pandas_type)
    assert_eq(dafunc(arg1, arg2), npfunc(arg1, arg2))
    if ufunc == 'ldexp':
        return
    assert isinstance(dafunc(arg2, dask), dask_type)
    assert_eq(dafunc(arg2, dask), npfunc(arg2, dask))
    assert isinstance(npfunc(arg2, dask), dask_type)
    assert_eq(npfunc(arg2, dask), npfunc(arg2, dask))
    assert isinstance(dafunc(arg2, arg1), pandas_type)
    assert_eq(dafunc(arg2, arg1), npfunc(arg2, arg1))