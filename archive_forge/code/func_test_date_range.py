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
@pytest.mark.parametrize('start,calendar,use_cftime,expected_type', [('1990-01-01', 'standard', None, pd.DatetimeIndex), ('1990-01-01', 'proleptic_gregorian', True, CFTimeIndex), ('1990-01-01', 'noleap', None, CFTimeIndex), ('1990-01-01', 'gregorian', False, pd.DatetimeIndex), ('1400-01-01', 'standard', None, CFTimeIndex), ('3400-01-01', 'standard', None, CFTimeIndex)])
def test_date_range(start: str, calendar: str, use_cftime: bool | None, expected_type) -> None:
    dr = date_range(start, periods=14, freq='D', calendar=calendar, use_cftime=use_cftime)
    assert isinstance(dr, expected_type)