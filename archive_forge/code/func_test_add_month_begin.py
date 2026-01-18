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
@pytest.mark.parametrize(('initial_date_args', 'offset', 'expected_date_args'), [((1, 1, 1), MonthBegin(), (1, 2, 1)), ((1, 1, 1), MonthBegin(n=2), (1, 3, 1)), ((1, 1, 7), MonthBegin(), (1, 2, 1)), ((1, 1, 7), MonthBegin(n=2), (1, 3, 1)), ((1, 3, 1), MonthBegin(n=-1), (1, 2, 1)), ((1, 3, 1), MonthBegin(n=-2), (1, 1, 1)), ((1, 3, 3), MonthBegin(n=-1), (1, 3, 1)), ((1, 3, 3), MonthBegin(n=-2), (1, 2, 1)), ((1, 2, 1), MonthBegin(n=14), (2, 4, 1)), ((2, 4, 1), MonthBegin(n=-14), (1, 2, 1)), ((1, 1, 1, 5, 5, 5, 5), MonthBegin(), (1, 2, 1, 5, 5, 5, 5)), ((1, 1, 3, 5, 5, 5, 5), MonthBegin(), (1, 2, 1, 5, 5, 5, 5)), ((1, 1, 3, 5, 5, 5, 5), MonthBegin(n=-1), (1, 1, 1, 5, 5, 5, 5))], ids=_id_func)
def test_add_month_begin(calendar, initial_date_args, offset, expected_date_args):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    expected = date_type(*expected_date_args)
    assert result == expected