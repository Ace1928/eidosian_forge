from __future__ import annotations
import numpy as np
import pytest
from xarray import DataArray, infer_freq
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftime_offsets import date_range
from xarray.testing import assert_identical
from xarray.tests import requires_cftime
def test_convert_calendar_same_calendar():
    src = DataArray(date_range('2000-01-01', periods=12, freq='6h', use_cftime=False), dims=('time',), name='time')
    out = convert_calendar(src, 'proleptic_gregorian')
    assert src is out