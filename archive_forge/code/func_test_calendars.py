from __future__ import annotations
import datetime
from typing import TypedDict
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray.coding.cftime_offsets import _new_to_legacy_freq
from xarray.core.pdcompat import _convert_base_to_offset
from xarray.core.resample_cftime import CFTimeGrouper
@pytest.mark.filterwarnings('ignore:Converting a CFTimeIndex')
@pytest.mark.filterwarnings('ignore:.*the `(base|loffset)` parameter to resample')
@pytest.mark.parametrize('calendar', ['gregorian', 'noleap', 'all_leap', '360_day', 'julian'])
def test_calendars(calendar: str) -> None:
    freq, closed, label, base = ('8001min', None, None, 17)
    loffset = datetime.timedelta(hours=12)
    xr_index = xr.cftime_range(start='2004-01-01T12:07:01', periods=7, freq='3D', calendar=calendar)
    pd_index = pd.date_range(start='2004-01-01T12:07:01', periods=7, freq='3D')
    da_cftime = da(xr_index).resample(time=freq, closed=closed, label=label, base=base, loffset=loffset).mean()
    da_datetime = da(pd_index).resample(time=freq, closed=closed, label=label, base=base, loffset=loffset).mean()
    da_cftime['time'] = da_cftime.xindexes['time'].to_pandas_index().to_datetimeindex()
    xr.testing.assert_identical(da_cftime, da_datetime)