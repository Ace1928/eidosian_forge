from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_findall_multi_single_case(dtype) -> None:
    pat_str = '(\\w+)_Xy_(\\d*)'
    pat_re = re.compile(dtype(pat_str))
    value = xr.DataArray([['a_Xy_0', 'ab_xY_10', 'abc_Xy_01'], ['abcd_Xy_', '', 'abcdef_Xy_101']], dims=['X', 'Y']).astype(dtype)
    expected_list: list[list[list[list]]] = [[[['a', '0']], [], [['abc', '01']]], [[['abcd', '']], [], [['abcdef', '101']]]]
    expected_dtype = [[[tuple((dtype(x) for x in y)) for y in z] for z in w] for w in expected_list]
    expected_np = np.array(expected_dtype, dtype=np.object_)
    expected = xr.DataArray(expected_np, dims=['X', 'Y'])
    res_str = value.str.findall(pat=pat_str)
    res_re = value.str.findall(pat=pat_re)
    res_str_case = value.str.findall(pat=pat_str, case=True)
    assert res_str.dtype == expected.dtype
    assert res_re.dtype == expected.dtype
    assert res_str_case.dtype == expected.dtype
    assert_equal(res_str, expected)
    assert_equal(res_re, expected)
    assert_equal(res_str_case, expected)