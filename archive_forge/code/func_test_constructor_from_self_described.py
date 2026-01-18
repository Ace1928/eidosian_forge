from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_constructor_from_self_described(self) -> None:
    data = [[-0.1, 21], [0, 2]]
    expected = DataArray(data, coords={'x': ['a', 'b'], 'y': [-1, -2]}, dims=['x', 'y'], name='foobar', attrs={'bar': 2})
    actual = DataArray(expected)
    assert_identical(expected, actual)
    actual = DataArray(expected.values, actual.coords)
    assert_equal(expected, actual)
    frame = pd.DataFrame(data, index=pd.Index(['a', 'b'], name='x'), columns=pd.Index([-1, -2], name='y'))
    actual = DataArray(frame)
    assert_equal(expected, actual)
    series = pd.Series(data[0], index=pd.Index([-1, -2], name='y'))
    actual = DataArray(series)
    assert_equal(expected[0].reset_coords('x', drop=True), actual)
    expected = DataArray(data, coords={'x': ['a', 'b'], 'y': [-1, -2], 'a': 0, 'z': ('x', [-0.5, 0.5])}, dims=['x', 'y'])
    actual = DataArray(expected)
    assert_identical(expected, actual)
    actual = DataArray(expected.values, expected.coords)
    assert_identical(expected, actual)
    expected = Dataset({'foo': ('foo', ['a', 'b'])})['foo']
    actual = DataArray(pd.Index(['a', 'b'], name='foo'))
    assert_identical(expected, actual)
    actual = DataArray(IndexVariable('foo', ['a', 'b']))
    assert_identical(expected, actual)