from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_case_bytes() -> None:
    value = xr.DataArray(['SOme wOrd']).astype(np.bytes_)
    exp_capitalized = xr.DataArray(['Some word']).astype(np.bytes_)
    exp_lowered = xr.DataArray(['some word']).astype(np.bytes_)
    exp_swapped = xr.DataArray(['soME WoRD']).astype(np.bytes_)
    exp_titled = xr.DataArray(['Some Word']).astype(np.bytes_)
    exp_uppered = xr.DataArray(['SOME WORD']).astype(np.bytes_)
    res_capitalized = value.str.capitalize()
    res_lowered = value.str.lower()
    res_swapped = value.str.swapcase()
    res_titled = value.str.title()
    res_uppered = value.str.upper()
    assert res_capitalized.dtype == exp_capitalized.dtype
    assert res_lowered.dtype == exp_lowered.dtype
    assert res_swapped.dtype == exp_swapped.dtype
    assert res_titled.dtype == exp_titled.dtype
    assert res_uppered.dtype == exp_uppered.dtype
    assert_equal(res_capitalized, exp_capitalized)
    assert_equal(res_lowered, exp_lowered)
    assert_equal(res_swapped, exp_swapped)
    assert_equal(res_titled, exp_titled)
    assert_equal(res_uppered, exp_uppered)