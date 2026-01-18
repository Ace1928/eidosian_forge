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
@requires_cftime
@pytest.mark.parametrize('freq', ('YE', 'YS', 'YE-MAY', 'MS', 'ME', 'QS', 'h', 'min', 's'))
@pytest.mark.parametrize('use_cftime', (True, False))
def test_date_range_like_no_deprecation(freq, use_cftime):
    source = date_range('2000', periods=3, freq=freq, use_cftime=False)
    with assert_no_warnings():
        date_range_like(source, 'standard', use_cftime=use_cftime)