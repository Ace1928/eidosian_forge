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
@requires_scipy
def test_datetime_single_string() -> None:
    da = xr.DataArray(np.arange(24), dims='time', coords={'time': pd.date_range('2000-01-01', periods=24)})
    actual = da.interp(time='2000-01-01T12:00')
    expected = xr.DataArray(0.5)
    assert_allclose(actual.drop_vars('time'), expected)