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
def test_cftimeindex_sub_cftimeindex(calendar) -> None:
    a = xr.cftime_range('2000', periods=5, calendar=calendar)
    b = a.shift(2, 'D')
    result = b - a
    expected = pd.TimedeltaIndex([timedelta(days=2) for _ in range(5)])
    assert result.equals(expected)
    assert isinstance(result, pd.TimedeltaIndex)