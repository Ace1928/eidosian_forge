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
def test_concat_data_vars_typing(self) -> None:
    data = Dataset({'foo': ('x', np.random.randn(10))})
    objs: list[Dataset] = [data.isel(x=slice(5)), data.isel(x=slice(5, None))]
    actual = concat(objs, dim='x', data_vars='minimal')
    assert_identical(data, actual)