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
@pytest.mark.parametrize('periods,expected', [(2, f"CFTimeIndex([2000-01-01 00:00:00, 2000-01-02 00:00:00],\n            dtype='object', length=2, calendar='{standard_or_gregorian}', freq=None)"), (4, f"CFTimeIndex([2000-01-01 00:00:00, 2000-01-02 00:00:00, 2000-01-03 00:00:00,\n             2000-01-04 00:00:00],\n            dtype='object', length=4, calendar='{standard_or_gregorian}', freq='D')"), (101, f"CFTimeIndex([2000-01-01 00:00:00, 2000-01-02 00:00:00, 2000-01-03 00:00:00,\n             2000-01-04 00:00:00, 2000-01-05 00:00:00, 2000-01-06 00:00:00,\n             2000-01-07 00:00:00, 2000-01-08 00:00:00, 2000-01-09 00:00:00,\n             2000-01-10 00:00:00,\n             ...\n             2000-04-01 00:00:00, 2000-04-02 00:00:00, 2000-04-03 00:00:00,\n             2000-04-04 00:00:00, 2000-04-05 00:00:00, 2000-04-06 00:00:00,\n             2000-04-07 00:00:00, 2000-04-08 00:00:00, 2000-04-09 00:00:00,\n             2000-04-10 00:00:00],\n            dtype='object', length=101, calendar='{standard_or_gregorian}', freq='D')")])
def test_cftimeindex_repr_formatting(periods, expected):
    """Test that cftimeindex.__repr__ is formatted similar to pd.Index.__repr__."""
    index = xr.cftime_range(start='2000', periods=periods, freq='D')
    expected = dedent(expected)
    assert expected == repr(index)