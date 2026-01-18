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
def test_interp1d_bounds_error() -> None:
    """Ensure exception on bounds error is raised if requested"""
    da = xr.DataArray(np.sin(0.3 * np.arange(4)), [('time', np.arange(4))])
    with pytest.raises(ValueError):
        da.interp(time=3.5, kwargs=dict(bounds_error=True))
    da.interp(time=3.5)