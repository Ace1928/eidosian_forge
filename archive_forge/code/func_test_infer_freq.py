from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
@pytest.mark.parametrize('freq', ['300YS-JAN', 'YE-DEC', 'YS-JUL', '2YS-FEB', 'QE-NOV', '3QS-DEC', 'MS', '4ME', '7D', 'D', '30h', '5min', '40s'])
@pytest.mark.parametrize('calendar', _CFTIME_CALENDARS)
def test_infer_freq(freq, calendar):
    indx = xr.cftime_range('2000-01-01', periods=3, freq=freq, calendar=calendar)
    out = xr.infer_freq(indx)
    assert out == freq