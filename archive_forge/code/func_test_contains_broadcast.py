from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_contains_broadcast(dtype) -> None:
    values = xr.DataArray(['Foo', 'xYz', 'fOOomMm__fOo', 'MMM_'], dims='X').astype(dtype)
    pat_str = xr.DataArray(['FOO|mmm', 'Foo', 'MMM'], dims='Y').astype(dtype)
    pat_re = xr.DataArray([re.compile(x) for x in pat_str.data], dims='Y')
    result = values.str.contains(pat_str, case=False)
    expected = xr.DataArray([[True, True, False], [False, False, False], [True, True, True], [True, False, True]], dims=['X', 'Y'])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.contains(pat_str)
    expected = xr.DataArray([[False, True, False], [False, False, False], [False, False, False], [False, False, True]], dims=['X', 'Y'])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.contains(pat_re)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.contains(pat_str, regex=False, case=False)
    expected = xr.DataArray([[False, True, False], [False, False, False], [False, True, True], [False, False, True]], dims=['X', 'Y'])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.contains(pat_str, regex=False, case=True)
    expected = xr.DataArray([[False, True, False], [False, False, False], [False, False, False], [False, False, True]], dims=['X', 'Y'])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)