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
def test_concat_multiple_datasets_with_multiple_missing_variables() -> None:
    vars_to_drop_in_first = ['temperature', 'pressure']
    vars_to_drop_in_second = ['humidity', 'precipitation', 'cloud_cover']
    datasets = create_concat_datasets(2, seed=123)
    expected = concat(datasets, dim='day')
    for name in vars_to_drop_in_first:
        expected[name][..., :2] = np.nan
    for name in vars_to_drop_in_second:
        expected[name][..., 2:] = np.nan
    datasets[0] = datasets[0].drop_vars(vars_to_drop_in_first)
    datasets[1] = datasets[1].drop_vars(vars_to_drop_in_second)
    actual = concat(datasets, dim='day')
    assert list(actual.data_vars.keys()) == ['humidity', 'precipitation', 'cloud_cover', 'temperature', 'pressure']
    assert_identical(actual, expected)