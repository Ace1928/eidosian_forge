from __future__ import annotations
from itertools import combinations, permutations
from typing import cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import _parse_array_of_cftime_strings
from xarray.core.types import InterpOptions
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
@requires_cftime
@requires_scipy
def test_cftime() -> None:
    times = xr.cftime_range('2000', periods=24, freq='D')
    da = xr.DataArray(np.arange(24), coords=[times], dims='time')
    times_new = xr.cftime_range('2000-01-01T12:00:00', periods=3, freq='D')
    actual = da.interp(time=times_new)
    expected = xr.DataArray([0.5, 1.5, 2.5], coords=[times_new], dims=['time'])
    assert_allclose(actual, expected)