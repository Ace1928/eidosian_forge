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
def test_getitem_dataarray(self) -> None:
    da = DataArray(np.arange(12).reshape((3, 4)), dims=['x', 'y'])
    ind = DataArray([[0, 1], [0, 1]], dims=['x', 'z'])
    actual = da[ind]
    assert_array_equal(actual, da.values[[[0, 1], [0, 1]], :])
    da = DataArray(np.arange(12).reshape((3, 4)), dims=['x', 'y'], coords={'x': [0, 1, 2], 'y': ['a', 'b', 'c', 'd']})
    ind = xr.DataArray([[0, 1], [0, 1]], dims=['X', 'Y'])
    actual = da[ind]
    expected = da.values[[[0, 1], [0, 1]], :]
    assert_array_equal(actual, expected)
    assert actual.dims == ('X', 'Y', 'y')
    ind = xr.DataArray([True, True, False], dims=['x'])
    assert_equal(da[ind], da[[0, 1], :])
    assert_equal(da[ind], da[[0, 1]])
    assert_equal(da[ind], da[ind.values])