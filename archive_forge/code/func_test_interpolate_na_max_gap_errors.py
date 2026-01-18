from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
def test_interpolate_na_max_gap_errors(da_time):
    with pytest.raises(NotImplementedError, match='max_gap not implemented for unlabeled coordinates'):
        da_time.interpolate_na('t', max_gap=1)
    with pytest.raises(ValueError, match='max_gap must be a scalar.'):
        da_time.interpolate_na('t', max_gap=(1,))
    da_time['t'] = pd.date_range('2001-01-01', freq='h', periods=11)
    with pytest.raises(TypeError, match='Expected value of type str'):
        da_time.interpolate_na('t', max_gap=1)
    with pytest.raises(TypeError, match='Expected integer or floating point'):
        da_time.interpolate_na('t', max_gap='1h', use_coordinate=False)
    with pytest.raises(ValueError, match="Could not convert 'huh' to timedelta64"):
        da_time.interpolate_na('t', max_gap='huh')