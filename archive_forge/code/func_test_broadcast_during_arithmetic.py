from __future__ import annotations
import pytest
import xarray as xr
from xarray.testing import assert_equal
def test_broadcast_during_arithmetic(arrays: tuple[xr.DataArray, xr.DataArray]) -> None:
    np_arr, xp_arr = arrays
    np_arr2 = xr.DataArray(np.array([1.0, 2.0]), dims='x')
    xp_arr2 = xr.DataArray(xp.asarray([1.0, 2.0]), dims='x')
    expected = np_arr * np_arr2
    actual = xp_arr * xp_arr2
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)
    expected = np_arr2 * np_arr
    actual = xp_arr2 * xp_arr
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)