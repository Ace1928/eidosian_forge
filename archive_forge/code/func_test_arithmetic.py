from __future__ import annotations
import pytest
import xarray as xr
from xarray.testing import assert_equal
def test_arithmetic(arrays: tuple[xr.DataArray, xr.DataArray]) -> None:
    np_arr, xp_arr = arrays
    expected = np_arr + 7
    actual = xp_arr + 7
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)