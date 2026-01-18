from __future__ import annotations
import numpy as np
import pytest
from xarray import DataArray, infer_freq
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftime_offsets import date_range
from xarray.testing import assert_identical
from xarray.tests import requires_cftime
@pytest.mark.parametrize('source,target,freq', [('standard', '360_day', 'D'), ('360_day', 'proleptic_gregorian', 'D'), ('proleptic_gregorian', '360_day', '4h')])
@pytest.mark.parametrize('align_on', ['date', 'year'])
def test_convert_calendar_360_days(source, target, freq, align_on):
    src = DataArray(date_range('2004-01-01', '2004-12-30', freq=freq, calendar=source), dims=('time',), name='time')
    da_src = DataArray(np.linspace(0, 1, src.size), dims=('time',), coords={'time': src})
    conv = convert_calendar(da_src, target, align_on=align_on)
    assert conv.time.dt.calendar == target
    if align_on == 'date':
        np.testing.assert_array_equal(conv.time.resample(time='ME').last().dt.day, [30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30])
    elif target == '360_day':
        np.testing.assert_array_equal(conv.time.resample(time='ME').last().dt.day, [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29])
    else:
        np.testing.assert_array_equal(conv.time.resample(time='ME').last().dt.day, [30, 29, 30, 30, 31, 30, 30, 31, 30, 31, 29, 31])
    if source == '360_day' and align_on == 'year':
        assert conv.size == 360 if freq == 'D' else 360 * 4
    else:
        assert conv.size == 359 if freq == 'D' else 359 * 4