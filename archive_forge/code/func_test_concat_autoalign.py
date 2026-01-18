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
def test_concat_autoalign(self) -> None:
    ds1 = Dataset({'foo': DataArray([1, 2], coords=[('x', [1, 2])])})
    ds2 = Dataset({'foo': DataArray([1, 2], coords=[('x', [1, 3])])})
    actual = concat([ds1, ds2], 'y')
    expected = Dataset({'foo': DataArray([[1, 2, np.nan], [1, np.nan, 2]], dims=['y', 'x'], coords={'x': [1, 2, 3]})})
    assert_identical(expected, actual)