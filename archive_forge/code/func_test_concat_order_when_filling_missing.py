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
def test_concat_order_when_filling_missing() -> None:
    vars_to_drop_in_first: list[str] = []
    vars_to_drop_in_second = ['humidity']
    datasets = create_concat_datasets(2, seed=123)
    expected1 = concat(datasets, dim='day')
    for name in vars_to_drop_in_second:
        expected1[name][..., 2:] = np.nan
    expected2 = concat(datasets[::-1], dim='day')
    for name in vars_to_drop_in_second:
        expected2[name][..., :2] = np.nan
    datasets[0] = datasets[0].drop_vars(vars_to_drop_in_first)
    datasets[1] = datasets[1].drop_vars(vars_to_drop_in_second)
    actual = concat(datasets, dim='day')
    assert list(actual.data_vars.keys()) == ['temperature', 'pressure', 'humidity', 'precipitation', 'cloud_cover']
    assert_identical(actual, expected1)
    actual = concat(datasets[::-1], dim='day')
    assert list(actual.data_vars.keys()) == ['temperature', 'pressure', 'precipitation', 'cloud_cover', 'humidity']
    assert_identical(actual, expected2)