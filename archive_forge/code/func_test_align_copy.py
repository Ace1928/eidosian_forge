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
def test_align_copy(self) -> None:
    x = DataArray([1, 2, 3], coords=[('a', [1, 2, 3])])
    y = DataArray([1, 2], coords=[('a', [3, 1])])
    expected_x2 = x
    expected_y2 = DataArray([2, np.nan, 1], coords=[('a', [1, 2, 3])])
    x2, y2 = align(x, y, join='outer', copy=False)
    assert_identical(expected_x2, x2)
    assert_identical(expected_y2, y2)
    assert source_ndarray(x2.data) is source_ndarray(x.data)
    x2, y2 = align(x, y, join='outer', copy=True)
    assert_identical(expected_x2, x2)
    assert_identical(expected_y2, y2)
    assert source_ndarray(x2.data) is not source_ndarray(x.data)
    x = DataArray([1, 2, 3], coords=[('a', [1, 2, 3])])
    x2, = align(x, copy=False)
    assert_identical(x, x2)
    assert source_ndarray(x2.data) is source_ndarray(x.data)
    x2, = align(x, copy=True)
    assert_identical(x, x2)
    assert source_ndarray(x2.data) is not source_ndarray(x.data)