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
def test_broadcast_arrays(self) -> None:
    x = DataArray([1, 2], coords=[('a', [-1, -2])], name='x')
    y = DataArray([1, 2], coords=[('b', [3, 4])], name='y')
    x2, y2 = broadcast(x, y)
    expected_coords = [('a', [-1, -2]), ('b', [3, 4])]
    expected_x2 = DataArray([[1, 1], [2, 2]], expected_coords, name='x')
    expected_y2 = DataArray([[1, 2], [1, 2]], expected_coords, name='y')
    assert_identical(expected_x2, x2)
    assert_identical(expected_y2, y2)
    x = DataArray(np.random.randn(2, 3), dims=['a', 'b'])
    y = DataArray(np.random.randn(3, 2), dims=['b', 'a'])
    x2, y2 = broadcast(x, y)
    expected_x2 = x
    expected_y2 = y.T
    assert_identical(expected_x2, x2)
    assert_identical(expected_y2, y2)