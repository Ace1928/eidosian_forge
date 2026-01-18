from __future__ import annotations
from itertools import product
from typing import Callable, Literal
import numpy as np
import pandas as pd
import pytest
from xarray import CFTimeIndex
from xarray.coding.cftime_offsets import (
from xarray.coding.frequencies import infer_freq
from xarray.core.dataarray import DataArray
from xarray.tests import (
@pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
def test_date_range_like_errors():
    src = date_range('1899-02-03', periods=20, freq='D', use_cftime=False)
    src = src[np.arange(20) != 10]
    with pytest.raises(ValueError, match='`date_range_like` was unable to generate a range as the source frequency was not inferable.'):
        date_range_like(src, 'gregorian')
    src = DataArray(np.array([['1999-01-01', '1999-01-02'], ['1999-01-03', '1999-01-04']], dtype=np.datetime64), dims=('x', 'y'))
    with pytest.raises(ValueError, match="'source' must be a 1D array of datetime objects for inferring its range."):
        date_range_like(src, 'noleap')
    da = DataArray([1, 2, 3, 4], dims=('time',))
    with pytest.raises(ValueError, match="'source' must be a 1D array of datetime objects for inferring its range."):
        date_range_like(da, 'noleap')