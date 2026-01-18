from __future__ import annotations
import numpy as np
import pytest
from xarray import DataArray, infer_freq
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftime_offsets import date_range
from xarray.testing import assert_identical
from xarray.tests import requires_cftime
@requires_cftime
@pytest.mark.parametrize('source,target,freq', [('standard', 'noleap', 'D'), ('noleap', 'proleptic_gregorian', '4h'), ('noleap', 'all_leap', 'ME'), ('360_day', 'noleap', 'D'), ('noleap', '360_day', 'D')])
def test_convert_calendar_missing(source, target, freq):
    src = DataArray(date_range('2004-01-01', '2004-12-31' if source != '360_day' else '2004-12-30', freq=freq, calendar=source), dims=('time',), name='time')
    da_src = DataArray(np.linspace(0, 1, src.size), dims=('time',), coords={'time': src})
    out = convert_calendar(da_src, target, missing=np.nan, align_on='date')
    expected_freq = freq
    assert infer_freq(out.time) == expected_freq
    expected = date_range('2004-01-01', '2004-12-31' if target != '360_day' else '2004-12-30', freq=freq, calendar=target)
    np.testing.assert_array_equal(out.time, expected)
    if freq != 'ME':
        out_without_missing = convert_calendar(da_src, target, align_on='date')
        expected_nan = out.isel(time=~out.time.isin(out_without_missing.time))
        assert expected_nan.isnull().all()
        expected_not_nan = out.sel(time=out_without_missing.time)
        assert_identical(expected_not_nan, out_without_missing)