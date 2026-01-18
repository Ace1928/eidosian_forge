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
@pytest.mark.parametrize(('initial_date_args', 'offset', 'expected_year_month', 'expected_sub_day'), [((1, 1, 1), MonthEnd(), (1, 1), ()), ((1, 1, 1), MonthEnd(n=2), (1, 2), ()), ((1, 3, 1), MonthEnd(n=-1), (1, 2), ()), ((1, 3, 1), MonthEnd(n=-2), (1, 1), ()), ((1, 2, 1), MonthEnd(n=14), (2, 3), ()), ((2, 4, 1), MonthEnd(n=-14), (1, 2), ()), ((1, 1, 1, 5, 5, 5, 5), MonthEnd(), (1, 1), (5, 5, 5, 5)), ((1, 2, 1, 5, 5, 5, 5), MonthEnd(n=-1), (1, 1), (5, 5, 5, 5))], ids=_id_func)
def test_add_month_end(calendar, initial_date_args, offset, expected_year_month, expected_sub_day):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    reference_args = expected_year_month + (1,)
    reference = date_type(*reference_args)
    expected_date_args = expected_year_month + (_days_in_month(reference),) + expected_sub_day
    expected = date_type(*expected_date_args)
    assert result == expected