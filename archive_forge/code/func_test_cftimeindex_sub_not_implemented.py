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
def test_cftimeindex_sub_not_implemented(calendar):
    a = xr.cftime_range('2000', periods=5, calendar=calendar)
    with pytest.raises(TypeError, match='unsupported operand'):
        a - 1