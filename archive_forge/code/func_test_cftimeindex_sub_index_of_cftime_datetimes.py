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
@pytest.mark.parametrize('calendar', _CFTIME_CALENDARS)
def test_cftimeindex_sub_index_of_cftime_datetimes(calendar):
    a = xr.cftime_range('2000', periods=5, calendar=calendar)
    b = pd.Index(a.values)
    expected = a - a
    result = a - b
    assert result.equals(expected)
    assert isinstance(result, pd.TimedeltaIndex)