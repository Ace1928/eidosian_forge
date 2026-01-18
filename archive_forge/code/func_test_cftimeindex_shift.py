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
@pytest.mark.parametrize('freq', ['D', timedelta(days=1)])
def test_cftimeindex_shift(index, freq) -> None:
    date_type = index.date_type
    expected_dates = [date_type(1, 1, 3), date_type(1, 2, 3), date_type(2, 1, 3), date_type(2, 2, 3)]
    expected = CFTimeIndex(expected_dates)
    result = index.shift(2, freq)
    assert result.equals(expected)
    assert isinstance(result, CFTimeIndex)