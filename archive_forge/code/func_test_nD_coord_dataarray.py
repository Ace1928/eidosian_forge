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
def test_nD_coord_dataarray() -> None:
    da = DataArray(np.ones((2, 4)), dims=('x', 'y'), coords={'x': (('x', 'y'), np.arange(8).reshape((2, 4))), 'y': ('y', np.arange(4))})
    _assert_internal_invariants(da, check_default_indexes=True)
    da2 = DataArray(np.ones(4), dims='y', coords={'y': ('y', np.arange(4))})
    da3 = DataArray(np.ones(4), dims='z')
    _, actual = xr.align(da, da2)
    assert_identical(da2, actual)
    expected = da.drop_vars('x')
    _, actual = xr.broadcast(da, da2)
    assert_identical(expected, actual)
    actual, _ = xr.broadcast(da, da3)
    expected = da.expand_dims(z=4, axis=-1)
    assert_identical(actual, expected)
    da4 = DataArray(np.ones((2, 4)), coords={'x': 0}, dims=['x', 'y'])
    _assert_internal_invariants(da4, check_default_indexes=True)
    assert 'x' not in da4.xindexes
    assert 'x' in da4.coords