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
def test_concat_all_empty() -> None:
    ds1 = Dataset()
    ds2 = Dataset()
    expected = Dataset()
    actual = concat([ds1, ds2], dim='new_dim')
    assert_identical(actual, expected)