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
def test_asi8_distant_date():
    """Test that asi8 conversion is truly exact."""
    import cftime
    date_type = cftime.DatetimeProlepticGregorian
    index = xr.CFTimeIndex([date_type(10731, 4, 22, 3, 25, 45, 123456)])
    result = index.asi8
    expected = np.array([1000000 * 86400 * 400 * 8000 + 12345 * 1000000 + 123456])
    np.testing.assert_array_equal(result, expected)