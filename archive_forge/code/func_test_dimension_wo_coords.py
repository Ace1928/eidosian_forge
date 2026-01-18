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
def test_dimension_wo_coords() -> None:
    da = xr.DataArray(np.arange(12).reshape(3, 4), dims=['x', 'y'], coords={'y': [0, 1, 2, 3]})
    da_w_coord = da.copy()
    da_w_coord['x'] = np.arange(3)
    assert_equal(da.interp(x=[0.1, 0.2, 0.3]), da_w_coord.interp(x=[0.1, 0.2, 0.3]))
    assert_equal(da.interp(x=[0.1, 0.2, 0.3], y=[0.5]), da_w_coord.interp(x=[0.1, 0.2, 0.3], y=[0.5]))