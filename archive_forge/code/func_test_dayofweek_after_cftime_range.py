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
@pytest.mark.parametrize('freq', ['YE', 'ME', 'D'])
def test_dayofweek_after_cftime_range(freq: str) -> None:
    result = cftime_range('2000-02-01', periods=3, freq=freq).dayofweek
    freq = _new_to_legacy_freq(freq)
    expected = pd.date_range('2000-02-01', periods=3, freq=freq).dayofweek
    np.testing.assert_array_equal(result, expected)