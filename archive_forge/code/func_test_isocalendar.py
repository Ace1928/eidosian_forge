from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@pytest.mark.parametrize('field, pandas_field', [('year', 'year'), ('week', 'week'), ('weekday', 'day')])
def test_isocalendar(self, field, pandas_field) -> None:
    expected = pd.Index(getattr(self.times.isocalendar(), pandas_field).astype(int))
    expected = xr.DataArray(expected, name=field, coords=[self.times], dims=['time'])
    actual = self.data.time.dt.isocalendar()[field]
    assert_equal(expected, actual)