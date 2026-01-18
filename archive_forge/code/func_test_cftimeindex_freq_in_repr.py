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
@pytest.mark.parametrize('calendar', ['noleap', '360_day', 'standard'])
@pytest.mark.parametrize('freq', ['D', 'h'])
def test_cftimeindex_freq_in_repr(freq, calendar):
    """Test that cftimeindex has frequency property in repr."""
    index = xr.cftime_range(start='2000', periods=3, freq=freq, calendar=calendar)
    repr_str = index.__repr__()
    assert f", freq='{freq}'" in repr_str