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
@pytest.mark.parametrize('calendar', ['noleap', '365_day', '360_day', 'julian', 'gregorian', 'standard', 'proleptic_gregorian'])
def test_cftimeindex_freq_property_none_size_lt_3(calendar):
    for periods in range(3):
        index = xr.cftime_range(start='2000', periods=periods, calendar=calendar)
        assert index.freq is None