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
def test_interp1d_complex_out_of_bounds() -> None:
    """Ensure complex nans are used by default"""
    da = xr.DataArray(np.exp(0.3j * np.arange(4)), [('time', np.arange(4))])
    expected = da.interp(time=3.5, kwargs=dict(fill_value=np.nan + np.nan * 1j))
    actual = da.interp(time=3.5)
    assert_identical(actual, expected)