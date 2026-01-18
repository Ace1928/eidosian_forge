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
@pytest.mark.parametrize(('initial_year_month', 'initial_sub_day', 'offset', 'expected_year_month', 'expected_sub_day'), [((1, 12), (), YearEnd(), (2, 12), ()), ((1, 12), (), YearEnd(n=2), (3, 12), ()), ((2, 12), (), YearEnd(n=-1), (1, 12), ()), ((3, 12), (), YearEnd(n=-2), (1, 12), ()), ((1, 1), (), YearEnd(month=2), (1, 2), ()), ((1, 12), (5, 5, 5, 5), YearEnd(), (2, 12), (5, 5, 5, 5)), ((2, 12), (5, 5, 5, 5), YearEnd(n=-1), (1, 12), (5, 5, 5, 5))], ids=_id_func)
def test_add_year_end_onOffset(calendar, initial_year_month, initial_sub_day, offset, expected_year_month, expected_sub_day):
    date_type = get_date_type(calendar)
    reference_args = initial_year_month + (1,)
    reference = date_type(*reference_args)
    initial_date_args = initial_year_month + (_days_in_month(reference),) + initial_sub_day
    initial = date_type(*initial_date_args)
    result = initial + offset
    reference_args = expected_year_month + (1,)
    reference = date_type(*reference_args)
    expected_date_args = expected_year_month + (_days_in_month(reference),) + expected_sub_day
    expected = date_type(*expected_date_args)
    assert result == expected