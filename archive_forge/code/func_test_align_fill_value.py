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
@pytest.mark.parametrize('fill_value', [dtypes.NA, 2, 2.0, {'foo': 2, 'bar': 1}])
def test_align_fill_value(self, fill_value) -> None:
    x = Dataset({'foo': DataArray([1, 2], dims=['x'], coords={'x': [1, 2]})})
    y = Dataset({'bar': DataArray([1, 2], dims=['x'], coords={'x': [1, 3]})})
    x2, y2 = align(x, y, join='outer', fill_value=fill_value)
    if fill_value == dtypes.NA:
        fill_value_foo = fill_value_bar = np.nan
    elif isinstance(fill_value, dict):
        fill_value_foo = fill_value['foo']
        fill_value_bar = fill_value['bar']
    else:
        fill_value_foo = fill_value_bar = fill_value
    expected_x2 = Dataset({'foo': DataArray([1, 2, fill_value_foo], dims=['x'], coords={'x': [1, 2, 3]})})
    expected_y2 = Dataset({'bar': DataArray([1, fill_value_bar, 2], dims=['x'], coords={'x': [1, 2, 3]})})
    assert_identical(expected_x2, x2)
    assert_identical(expected_y2, y2)