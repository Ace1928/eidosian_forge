from __future__ import annotations
import numpy as np
import pytest
from xarray import DataArray, infer_freq
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftime_offsets import date_range
from xarray.testing import assert_identical
from xarray.tests import requires_cftime
@requires_cftime
def test_interp_calendar_errors():
    src_nl = DataArray([1] * 100, dims=('time',), coords={'time': date_range('0000-01-01', periods=100, freq='MS', calendar='noleap')})
    tgt_360 = date_range('0001-01-01', '0001-12-30', freq='MS', calendar='standard')
    with pytest.raises(ValueError, match='Source time coordinate contains dates with year 0'):
        interp_calendar(src_nl, tgt_360)
    da1 = DataArray([0, 1, 2], dims=('x',), name='x')
    da2 = da1 + 1
    with pytest.raises(ValueError, match="Both 'source.x' and 'target' must contain datetime objects."):
        interp_calendar(da1, da2, dim='x')