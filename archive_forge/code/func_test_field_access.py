from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@pytest.mark.parametrize('field', ['days', 'seconds', 'microseconds', 'nanoseconds'])
def test_field_access(self, field) -> None:
    expected = xr.DataArray(getattr(self.times, field), name=field, coords=[self.times], dims=['time'])
    actual = getattr(self.data.time.dt, field)
    assert_equal(expected, actual)