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
def test_cftimeindex_sub_timedeltaindex(calendar) -> None:
    a = xr.cftime_range('2000', periods=5, calendar=calendar)
    deltas = pd.TimedeltaIndex([timedelta(days=2) for _ in range(5)])
    result = a - deltas
    expected = a.shift(-2, 'D')
    assert result.equals(expected)
    assert isinstance(result, CFTimeIndex)