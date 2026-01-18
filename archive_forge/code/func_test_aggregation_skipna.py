from __future__ import annotations
import pytest
import xarray as xr
from xarray.testing import assert_equal
def test_aggregation_skipna(arrays) -> None:
    np_arr, xp_arr = arrays
    expected = np_arr.sum(skipna=False)
    actual = xp_arr.sum(skipna=False)
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)