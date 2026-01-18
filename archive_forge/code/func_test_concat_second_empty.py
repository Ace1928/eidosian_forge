from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import PandasIndex
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
def test_concat_second_empty() -> None:
    ds1 = Dataset(data_vars={'a': ('y', [0.1])}, coords={'x': 0.1})
    ds2 = Dataset(coords={'x': 0.1})
    expected = Dataset(data_vars={'a': ('y', [0.1, np.nan])}, coords={'x': 0.1})
    actual = concat([ds1, ds2], dim='y')
    assert_identical(actual, expected)
    expected = Dataset(data_vars={'a': ('y', [0.1, np.nan])}, coords={'x': ('y', [0.1, 0.1])})
    actual = concat([ds1, ds2], dim='y', coords='all')
    assert_identical(actual, expected)
    ds1['b'] = 0.1
    expected = Dataset(data_vars={'a': ('y', [0.1, np.nan]), 'b': ('y', [0.1, np.nan])}, coords={'x': ('y', [0.1, 0.1])})
    actual = concat([ds1, ds2], dim='y', coords='all', data_vars='all')
    assert_identical(actual, expected)
    expected = Dataset(data_vars={'a': ('y', [0.1, np.nan]), 'b': 0.1}, coords={'x': 0.1})
    actual = concat([ds1, ds2], dim='y', coords='different', data_vars='different')
    assert_identical(actual, expected)