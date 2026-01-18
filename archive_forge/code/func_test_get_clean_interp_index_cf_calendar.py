from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_cftime
@pytest.mark.parametrize('calendar', _CFTIME_CALENDARS)
def test_get_clean_interp_index_cf_calendar(cf_da, calendar):
    """The index for CFTimeIndex is in units of days. This means that if two series using a 360 and 365 days
    calendar each have a trend of .01C/year, the linear regression coefficients will be different because they
    have different number of days.

    Another option would be to have an index in units of years, but this would likely create other difficulties.
    """
    i = get_clean_interp_index(cf_da(calendar), dim='time')
    np.testing.assert_array_equal(i, np.arange(10) * 1000000000.0 * 86400)