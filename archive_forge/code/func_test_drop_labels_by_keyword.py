from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_drop_labels_by_keyword(self) -> None:
    data = Dataset({'A': (['x', 'y'], np.random.randn(2, 6)), 'x': ['a', 'b'], 'y': range(6)})
    assert len(data.coords['x']) == 2
    with pytest.warns(DeprecationWarning):
        ds1 = data.drop(['a'], dim='x')
    ds2 = data.drop_sel(x='a')
    ds3 = data.drop_sel(x=['a'])
    ds4 = data.drop_sel(x=['a', 'b'])
    ds5 = data.drop_sel(x=['a', 'b'], y=range(0, 6, 2))
    arr = DataArray(range(3), dims=['c'])
    with pytest.warns(DeprecationWarning):
        data.drop(arr.coords)
    with pytest.warns(DeprecationWarning):
        data.drop(arr.xindexes)
    assert_array_equal(ds1.coords['x'], ['b'])
    assert_array_equal(ds2.coords['x'], ['b'])
    assert_array_equal(ds3.coords['x'], ['b'])
    assert ds4.coords['x'].size == 0
    assert ds5.coords['x'].size == 0
    assert_array_equal(ds5.coords['y'], [1, 3, 5])
    with pytest.raises(ValueError):
        data.drop(labels=['a'], x='a')
    with pytest.raises(ValueError):
        data.drop(labels=['a'], dim='x', x='a')
    warnings.filterwarnings('ignore', '\\W*drop')
    with pytest.raises(ValueError):
        data.drop(dim='x', x='a')