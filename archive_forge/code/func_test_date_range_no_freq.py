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
@pytest.mark.parametrize('start, end, periods', [('2022-01-01', '2022-01-10', 2), ('2022-03-01', '2022-03-31', 2), ('2022-01-01', '2022-01-10', None), ('2022-03-01', '2022-03-31', None)])
def test_date_range_no_freq(start, end, periods):
    """
    Test whether date_range produces the same result as Pandas
    when freq is not provided, but start, end and periods are.
    """
    result = date_range(start=start, end=end, periods=periods)
    expected = pd.date_range(start=start, end=end, periods=periods)
    np.testing.assert_array_equal(result, expected)