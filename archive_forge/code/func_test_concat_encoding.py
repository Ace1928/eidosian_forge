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
def test_concat_encoding(self) -> None:
    ds = Dataset({'foo': (['x', 'y'], np.random.random((2, 3))), 'bar': (['x', 'y'], np.random.random((2, 3)))}, {'x': [0, 1]})
    foo = ds['foo']
    foo.encoding = {'complevel': 5}
    ds.encoding = {'unlimited_dims': 'x'}
    assert concat([foo, foo], dim='x').encoding == foo.encoding
    assert concat([ds, ds], dim='x').encoding == ds.encoding