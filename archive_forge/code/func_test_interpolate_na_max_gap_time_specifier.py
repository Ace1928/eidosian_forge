from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_bottleneck
@pytest.mark.parametrize('time_range_func', [pd.date_range, pytest.param(xr.cftime_range, marks=requires_cftime)])
@pytest.mark.parametrize('transform', [lambda x: x, lambda x: x.to_dataset(name='a')])
@pytest.mark.parametrize('max_gap', ['3h', np.timedelta64(3, 'h'), pd.to_timedelta('3h')])
def test_interpolate_na_max_gap_time_specifier(da_time, max_gap, transform, time_range_func):
    da_time['t'] = time_range_func('2001-01-01', freq='h', periods=11)
    expected = transform(da_time.copy(data=[np.nan, 1, 2, 3, 4, 5, np.nan, np.nan, np.nan, np.nan, 10]))
    actual = transform(da_time).interpolate_na('t', max_gap=max_gap)
    assert_allclose(actual, expected)