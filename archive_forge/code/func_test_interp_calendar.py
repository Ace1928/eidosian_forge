from __future__ import annotations
import numpy as np
import pytest
from xarray import DataArray, infer_freq
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftime_offsets import date_range
from xarray.testing import assert_identical
from xarray.tests import requires_cftime
@pytest.mark.parametrize('source,target', [('standard', 'noleap'), ('noleap', 'proleptic_gregorian'), ('standard', '360_day'), ('360_day', 'proleptic_gregorian'), ('noleap', 'all_leap'), ('360_day', 'noleap')])
def test_interp_calendar(source, target):
    src = DataArray(date_range('2004-01-01', '2004-07-30', freq='D', calendar=source), dims=('time',), name='time')
    tgt = DataArray(date_range('2004-01-01', '2004-07-30', freq='D', calendar=target), dims=('time',), name='time')
    da_src = DataArray(np.linspace(0, 1, src.size), dims=('time',), coords={'time': src})
    conv = interp_calendar(da_src, tgt)
    assert_identical(tgt.time, conv.time)
    np.testing.assert_almost_equal(conv.max(), 1, 2)
    assert conv.min() == 0