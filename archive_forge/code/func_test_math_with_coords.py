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
def test_math_with_coords(self) -> None:
    coords = {'x': [-1, -2], 'y': ['ab', 'cd', 'ef'], 'lat': (['x', 'y'], [[1, 2, 3], [-1, -2, -3]]), 'c': -999}
    orig = DataArray(np.random.randn(2, 3), coords, dims=['x', 'y'])
    actual = orig + 1
    expected = DataArray(orig.values + 1, orig.coords)
    assert_identical(expected, actual)
    actual = 1 + orig
    assert_identical(expected, actual)
    actual = orig + orig[0, 0]
    exp_coords = {k: v for k, v in coords.items() if k != 'lat'}
    expected = DataArray(orig.values + orig.values[0, 0], exp_coords, dims=['x', 'y'])
    assert_identical(expected, actual)
    actual = orig[0, 0] + orig
    assert_identical(expected, actual)
    actual = orig[0, 0] + orig[-1, -1]
    expected = DataArray(orig.values[0, 0] + orig.values[-1, -1], {'c': -999})
    assert_identical(expected, actual)
    actual = orig[:, 0] + orig[0, :]
    exp_values = orig[:, 0].values[:, None] + orig[0, :].values[None, :]
    expected = DataArray(exp_values, exp_coords, dims=['x', 'y'])
    assert_identical(expected, actual)
    actual = orig[0, :] + orig[:, 0]
    assert_identical(expected.transpose(transpose_coords=True), actual)
    actual = orig - orig.transpose(transpose_coords=True)
    expected = DataArray(np.zeros((2, 3)), orig.coords)
    assert_identical(expected, actual)
    actual = orig.transpose(transpose_coords=True) - orig
    assert_identical(expected.transpose(transpose_coords=True), actual)
    alt = DataArray([1, 1], {'x': [-1, -2], 'c': 'foo', 'd': 555}, 'x')
    actual = orig + alt
    expected = orig + 1
    expected.coords['d'] = 555
    del expected.coords['c']
    assert_identical(expected, actual)
    actual = alt + orig
    assert_identical(expected, actual)