from __future__ import annotations
import random
import sys
from copy import deepcopy
from itertools import product
import numpy as np
import pytest
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, ComplexWarning
from dask.array.utils import assert_eq
from dask.base import tokenize
from dask.utils import typename
@pytest.mark.parametrize('funcname', ['ones_like', 'zeros_like', 'empty_like'])
def test_like_funcs(funcname):
    mask = np.array([[True, False], [True, True], [False, True]])
    data = np.arange(6).reshape((3, 2))
    a = np.ma.array(data, mask=mask)
    d_a = da.ma.masked_array(data=data, mask=mask, chunks=2)
    da_func = getattr(da.ma, funcname)
    np_func = getattr(np.ma.core, funcname)
    res = da_func(d_a)
    sol = np_func(a)
    if 'empty' in funcname:
        assert_eq(da.ma.getmaskarray(res), np.ma.getmaskarray(sol))
    else:
        assert_eq(res, sol)