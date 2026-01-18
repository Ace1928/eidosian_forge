from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
@pytest.mark.parametrize('func,sparse_output', [(do('squeeze'), True), param(do('to_index'), False, marks=xfail(reason='Coercion to dense')), param(do('to_index_variable'), False, marks=xfail(reason='Coercion to dense')), param(do('searchsorted', 0.5), True, marks=xfail(reason="'COO' object has no attribute 'searchsorted'"))])
def test_1d_variable_method(func, sparse_output):
    var_s = make_xrvar({'x': 10})
    var_d = xr.Variable(var_s.dims, var_s.data.todense())
    ret_s = func(var_s)
    ret_d = func(var_d)
    if sparse_output:
        assert isinstance(ret_s.data, sparse.SparseArray)
        assert np.allclose(ret_s.data.todense(), ret_d.data)
    else:
        assert np.allclose(ret_s, ret_d)