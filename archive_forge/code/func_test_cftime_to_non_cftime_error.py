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
def test_cftime_to_non_cftime_error() -> None:
    times = xr.cftime_range('2000', periods=24, freq='D')
    da = xr.DataArray(np.arange(24), coords=[times], dims='time')
    with pytest.raises(TypeError):
        da.interp(time=0.5)