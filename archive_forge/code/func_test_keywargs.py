from __future__ import annotations
from itertools import combinations, permutations
from typing import cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import _parse_array_of_cftime_strings
from xarray.core.types import InterpOptions
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
def test_keywargs():
    if not has_scipy:
        pytest.skip('scipy is not installed.')
    da = get_example_data(0)
    assert_equal(da.interp(x=[0.5, 0.8]), da.interp({'x': [0.5, 0.8]}))