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
def test_concat_multiple_missing_variables() -> None:
    datasets = create_concat_datasets(2, seed=123)
    expected = concat(datasets, dim='day')
    vars_to_drop = ['pressure', 'cloud_cover']
    expected['pressure'][..., 2:] = np.nan
    expected['cloud_cover'][..., 2:] = np.nan
    datasets[1] = datasets[1].drop_vars(vars_to_drop)
    actual = concat(datasets, dim='day')
    assert list(actual.data_vars.keys()) == ['temperature', 'pressure', 'humidity', 'precipitation', 'cloud_cover']
    assert_identical(actual, expected)