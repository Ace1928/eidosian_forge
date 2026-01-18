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
def test_concat_preserve_coordinate_order() -> None:
    x = np.arange(0, 5)
    y = np.arange(0, 10)
    time = np.arange(0, 4)
    data = np.zeros((4, 10, 5), dtype=bool)
    ds1 = Dataset({'data': (['time', 'y', 'x'], data[0:2])}, coords={'time': time[0:2], 'y': y, 'x': x})
    ds2 = Dataset({'data': (['time', 'y', 'x'], data[2:4])}, coords={'time': time[2:4], 'y': y, 'x': x})
    expected = Dataset({'data': (['time', 'y', 'x'], data)}, coords={'time': time, 'y': y, 'x': x})
    actual = concat([ds1, ds2], dim='time')
    for act, exp in zip(actual.dims, expected.dims):
        assert act == exp
        assert actual.sizes[act] == expected.sizes[exp]
    for act, exp in zip(actual.coords, expected.coords):
        assert act == exp
        assert_identical(actual.coords[act], expected.coords[exp])