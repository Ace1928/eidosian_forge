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
def test_drop_coordinates(self) -> None:
    expected = DataArray(np.random.randn(2, 3), dims=['x', 'y'])
    arr = expected.copy()
    arr.coords['z'] = 2
    actual = arr.drop_vars('z')
    assert_identical(expected, actual)
    with pytest.raises(ValueError):
        arr.drop_vars('not found')
    actual = expected.drop_vars('not found', errors='ignore')
    assert_identical(actual, expected)
    with pytest.raises(ValueError, match='cannot be found'):
        arr.drop_vars('w')
    actual = expected.drop_vars('w', errors='ignore')
    assert_identical(actual, expected)
    renamed = arr.rename('foo')
    with pytest.raises(ValueError, match='cannot be found'):
        renamed.drop_vars('foo')
    actual = renamed.drop_vars('foo', errors='ignore')
    assert_identical(actual, renamed)