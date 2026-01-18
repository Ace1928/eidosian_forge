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
def test_3641() -> None:
    times = xr.cftime_range('0001', periods=3, freq='500YE')
    da = xr.DataArray(range(3), dims=['time'], coords=[times])
    da.interp(time=['0002-05-01'])