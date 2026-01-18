from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_cat_str(dtype) -> None:
    values_1 = xr.DataArray([['a', 'bb', 'cccc'], ['ddddd', 'eeee', 'fff']], dims=['X', 'Y']).astype(dtype)
    values_2 = '111'
    targ_blank = xr.DataArray([['a111', 'bb111', 'cccc111'], ['ddddd111', 'eeee111', 'fff111']], dims=['X', 'Y']).astype(dtype)
    targ_space = xr.DataArray([['a 111', 'bb 111', 'cccc 111'], ['ddddd 111', 'eeee 111', 'fff 111']], dims=['X', 'Y']).astype(dtype)
    targ_bars = xr.DataArray([['a||111', 'bb||111', 'cccc||111'], ['ddddd||111', 'eeee||111', 'fff||111']], dims=['X', 'Y']).astype(dtype)
    targ_comma = xr.DataArray([['a, 111', 'bb, 111', 'cccc, 111'], ['ddddd, 111', 'eeee, 111', 'fff, 111']], dims=['X', 'Y']).astype(dtype)
    res_blank = values_1.str.cat(values_2)
    res_add = values_1.str + values_2
    res_space = values_1.str.cat(values_2, sep=' ')
    res_bars = values_1.str.cat(values_2, sep='||')
    res_comma = values_1.str.cat(values_2, sep=', ')
    assert res_blank.dtype == targ_blank.dtype
    assert res_add.dtype == targ_blank.dtype
    assert res_space.dtype == targ_space.dtype
    assert res_bars.dtype == targ_bars.dtype
    assert res_comma.dtype == targ_comma.dtype
    assert_equal(res_blank, targ_blank)
    assert_equal(res_add, targ_blank)
    assert_equal(res_space, targ_space)
    assert_equal(res_bars, targ_bars)
    assert_equal(res_comma, targ_comma)