from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_findall_single_multi_nocase(dtype) -> None:
    pat_str = '(\\w+)_Xy_\\d*'
    pat_re = re.compile(dtype(pat_str), flags=re.I)
    value = xr.DataArray([['a_Xy_0', 'ab_xY_10-bab_Xy_110-baab_Xy_1100', 'abc_Xy_01-cbc_Xy_2210'], ['abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210', '', 'abcdef_Xy_101-fef_Xy_5543210']], dims=['X', 'Y']).astype(dtype)
    expected_list: list[list[list]] = [[['a'], ['ab', 'bab', 'baab'], ['abc', 'cbc']], [['abcd', 'dcd', 'dccd'], [], ['abcdef', 'fef']]]
    expected_dtype = [[[dtype(x) for x in y] for y in z] for z in expected_list]
    expected_np = np.array(expected_dtype, dtype=np.object_)
    expected = xr.DataArray(expected_np, dims=['X', 'Y'])
    res_str = value.str.findall(pat=pat_str, case=False)
    res_re = value.str.findall(pat=pat_re)
    assert res_str.dtype == expected.dtype
    assert res_re.dtype == expected.dtype
    assert_equal(res_str, expected)
    assert_equal(res_re, expected)