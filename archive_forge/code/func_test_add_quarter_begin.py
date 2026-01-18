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
@pytest.mark.parametrize(('initial_date_args', 'offset', 'expected_date_args'), [((1, 1, 1), QuarterBegin(), (1, 3, 1)), ((1, 1, 1), QuarterBegin(n=2), (1, 6, 1)), ((1, 1, 1), QuarterBegin(month=2), (1, 2, 1)), ((1, 1, 7), QuarterBegin(n=2), (1, 6, 1)), ((2, 2, 1), QuarterBegin(n=-1), (1, 12, 1)), ((1, 3, 2), QuarterBegin(n=-1), (1, 3, 1)), ((1, 1, 1, 5, 5, 5, 5), QuarterBegin(), (1, 3, 1, 5, 5, 5, 5)), ((2, 1, 1, 5, 5, 5, 5), QuarterBegin(n=-1), (1, 12, 1, 5, 5, 5, 5))], ids=_id_func)
def test_add_quarter_begin(calendar, initial_date_args, offset, expected_date_args):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    expected = date_type(*expected_date_args)
    assert result == expected