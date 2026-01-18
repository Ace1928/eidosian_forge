from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@pytest.mark.parametrize('func', ['mean', 'sum'])
@pytest.mark.parametrize('min_periods', [1, 10])
def test_cumulative(d, func, min_periods) -> None:
    result = getattr(d.cumulative('z', min_periods=min_periods), func)()
    expected = getattr(d.rolling(z=d['z'].size, min_periods=min_periods), func)()
    assert_identical(result, expected)
    result = getattr(d.cumulative(['z', 'x'], min_periods=min_periods), func)()
    expected = getattr(d.rolling(z=d['z'].size, x=d['x'].size, min_periods=min_periods), func)()
    assert_identical(result, expected)