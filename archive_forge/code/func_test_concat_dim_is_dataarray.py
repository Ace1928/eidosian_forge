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
def test_concat_dim_is_dataarray(self) -> None:
    objs = [Dataset({'x': 0}), Dataset({'x': 1})]
    coord = DataArray([3, 4], dims='y', attrs={'foo': 'bar'})
    expected = Dataset({'x': ('y', [0, 1]), 'y': coord})
    actual = concat(objs, coord)
    assert_identical(actual, expected)