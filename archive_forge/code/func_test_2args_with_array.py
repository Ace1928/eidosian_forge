from __future__ import annotations
import warnings
import pytest
import numpy as np
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
@pytest.mark.skipif(dd._dask_expr_enabled(), reason="doesn't work at the moment, all return not implemented")
@pytest.mark.parametrize('ufunc', _UFUNCS_2ARG)
@pytest.mark.parametrize('pandas,darray', [(pd.Series(np.random.randint(1, 100, size=(100,))), da.from_array(np.random.randint(1, 100, size=(100,)), chunks=(50,))), (pd.DataFrame(np.random.randint(1, 100, size=(20, 2)), columns=['A', 'B']), da.from_array(np.random.randint(1, 100, size=(20, 2)), chunks=(10, 2)))])
def test_2args_with_array(ufunc, pandas, darray):
    dafunc = getattr(da, ufunc)
    npfunc = getattr(np, ufunc)
    dask = dd.from_pandas(pandas, 2)
    dask_type = dask.__class__
    assert isinstance(dafunc(dask, darray), dask_type)
    assert isinstance(dafunc(darray, dask), dask_type)
    np.testing.assert_array_equal(dafunc(dask, darray).compute().values, npfunc(pandas.values, darray).compute())
    assert isinstance(npfunc(dask, darray), dask_type)
    assert isinstance(npfunc(darray, dask), dask_type)
    np.testing.assert_array_equal(npfunc(dask, darray).compute().values, npfunc(pandas.values, darray.compute()))
    np.testing.assert_array_equal(npfunc(darray, dask).compute().values, npfunc(darray.compute(), pandas.values))