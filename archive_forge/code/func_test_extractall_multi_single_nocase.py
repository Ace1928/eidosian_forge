from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_extractall_multi_single_nocase(dtype) -> None:
    pat_str = '(\\w+)_Xy_(\\d*)'
    pat_re: str | bytes = pat_str if dtype == np.str_ else bytes(pat_str, encoding='UTF-8')
    pat_compiled = re.compile(pat_re, flags=re.I)
    value = xr.DataArray([['a_Xy_0', 'ab_xY_10', 'abc_Xy_01'], ['abcd_Xy_', '', 'abcdef_Xy_101']], dims=['X', 'Y']).astype(dtype)
    expected = xr.DataArray([[[['a', '0']], [['ab', '10']], [['abc', '01']]], [[['abcd', '']], [['', '']], [['abcdef', '101']]]], dims=['X', 'Y', 'XX', 'YY']).astype(dtype)
    res_str = value.str.extractall(pat=pat_str, group_dim='XX', match_dim='YY', case=False)
    res_re = value.str.extractall(pat=pat_compiled, group_dim='XX', match_dim='YY')
    assert res_str.dtype == expected.dtype
    assert res_re.dtype == expected.dtype
    assert_equal(res_str, expected)
    assert_equal(res_re, expected)