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
@pytest.mark.parametrize(('start', 'end', 'periods', 'freq', 'inclusive'), [(None, None, 5, 'YE', None), ('2000', None, None, 'YE', None), (None, '2000', None, 'YE', None), (None, None, None, None, None), ('2000', '2001', None, 'YE', 'up'), ('2000', '2001', 5, 'YE', None)])
def test_invalid_cftime_range_inputs(start: str | None, end: str | None, periods: int | None, freq: str | None, inclusive: Literal['up', None]) -> None:
    with pytest.raises(ValueError):
        cftime_range(start, end, periods, freq, inclusive=inclusive)