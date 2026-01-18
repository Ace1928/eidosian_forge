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
@pytest.mark.parametrize('function', [pytest.param(cftime_range, id='cftime', marks=requires_cftime), pytest.param(date_range, id='date')])
@pytest.mark.parametrize(('closed', 'inclusive'), [(None, 'both'), ('left', 'left'), ('right', 'right')])
def test_cftime_or_date_range_closed(function: Callable, closed: Literal['left', 'right', None], inclusive: Literal['left', 'right', 'both']) -> None:
    with pytest.warns(FutureWarning, match='Following pandas'):
        result_closed = function('2000-01-01', '2000-01-04', freq='D', closed=closed)
        result_inclusive = function('2000-01-01', '2000-01-04', freq='D', inclusive=inclusive)
        np.testing.assert_equal(result_closed.values, result_inclusive.values)