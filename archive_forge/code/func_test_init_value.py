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
def test_init_value(self) -> None:
    expected = DataArray(np.full((3, 4), 3), dims=['x', 'y'], coords=[range(3), range(4)])
    actual = DataArray(3, dims=['x', 'y'], coords=[range(3), range(4)])
    assert_identical(expected, actual)
    expected = DataArray(np.full((1, 10, 2), 0), dims=['w', 'x', 'y'], coords={'x': np.arange(10), 'y': ['north', 'south']})
    actual = DataArray(0, dims=expected.dims, coords=expected.coords)
    assert_identical(expected, actual)
    expected = DataArray(np.full((10, 2), np.nan), coords=[('x', np.arange(10)), ('y', ['a', 'b'])])
    actual = DataArray(coords=[('x', np.arange(10)), ('y', ['a', 'b'])])
    assert_identical(expected, actual)
    with pytest.raises(ValueError, match='different number of dim'):
        DataArray(np.array(1), coords={'x': np.arange(10)}, dims=['x'])
    with pytest.raises(ValueError, match='does not match the 0 dim'):
        DataArray(np.array(1), coords=[('x', np.arange(10))])