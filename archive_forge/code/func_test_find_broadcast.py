from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_find_broadcast(dtype) -> None:
    values = xr.DataArray(['ABCDEFG', 'BCDEFEF', 'DEFGHIJEF', 'EFGHEF', 'XXX'], dims=['X'])
    values = values.astype(dtype)
    sub = xr.DataArray(['EF', 'BC', 'XX'], dims=['Y']).astype(dtype)
    start = xr.DataArray([0, 7], dims=['Z'])
    end = xr.DataArray([6, 9], dims=['Z'])
    result_0 = values.str.find(sub, start, end)
    result_1 = values.str.find(sub, start, end, side='left')
    expected = xr.DataArray([[[4, -1], [1, -1], [-1, -1]], [[3, -1], [0, -1], [-1, -1]], [[1, 7], [-1, -1], [-1, -1]], [[0, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [0, -1]]], dims=['X', 'Y', 'Z'])
    assert result_0.dtype == expected.dtype
    assert result_1.dtype == expected.dtype
    assert_equal(result_0, expected)
    assert_equal(result_1, expected)
    result_0 = values.str.rfind(sub, start, end)
    result_1 = values.str.find(sub, start, end, side='right')
    expected = xr.DataArray([[[4, -1], [1, -1], [-1, -1]], [[3, -1], [0, -1], [-1, -1]], [[1, 7], [-1, -1], [-1, -1]], [[4, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [1, -1]]], dims=['X', 'Y', 'Z'])
    assert result_0.dtype == expected.dtype
    assert result_1.dtype == expected.dtype
    assert_equal(result_0, expected)
    assert_equal(result_1, expected)