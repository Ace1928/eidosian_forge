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
def test_concat_size0(self) -> None:
    data = create_test_data()
    split_data = [data.isel(dim1=slice(0, 0)), data]
    actual = concat(split_data, 'dim1')
    assert_identical(data, actual)
    actual = concat(split_data[::-1], 'dim1')
    assert_identical(data, actual)