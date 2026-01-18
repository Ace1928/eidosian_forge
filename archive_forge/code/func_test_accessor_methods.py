from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@pytest.mark.parametrize('method, parameters', [('floor', 'D'), ('ceil', 'D'), ('round', 'D')])
def test_accessor_methods(self, method, parameters) -> None:
    dates = pd.timedelta_range(start='1 day', end='30 days', freq='6h')
    xdates = xr.DataArray(dates, dims=['time'])
    expected = getattr(dates, method)(parameters)
    actual = getattr(xdates.dt, method)(parameters)
    assert_array_equal(expected, actual)