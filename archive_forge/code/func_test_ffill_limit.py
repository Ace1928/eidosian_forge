from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_bottleneck
def test_ffill_limit():
    da = xr.DataArray([0, np.nan, np.nan, np.nan, np.nan, 3, 4, 5, np.nan, 6, 7], dims='time')
    result = da.ffill('time')
    expected = xr.DataArray([0, 0, 0, 0, 0, 3, 4, 5, 5, 6, 7], dims='time')
    assert_array_equal(result, expected)
    result = da.ffill('time', limit=1)
    expected = xr.DataArray([0, 0, np.nan, np.nan, np.nan, 3, 4, 5, 5, 6, 7], dims='time')
    assert_array_equal(result, expected)