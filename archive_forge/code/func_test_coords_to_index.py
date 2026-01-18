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
def test_coords_to_index(self) -> None:
    da = DataArray(np.zeros((2, 3)), [('x', [1, 2]), ('y', list('abc'))])
    with pytest.raises(ValueError, match='no valid index'):
        da[0, 0].coords.to_index()
    expected = pd.Index(['a', 'b', 'c'], name='y')
    actual = da[0].coords.to_index()
    assert expected.equals(actual)
    expected = pd.MultiIndex.from_product([[1, 2], ['a', 'b', 'c']], names=['x', 'y'])
    actual = da.coords.to_index()
    assert expected.equals(actual)
    expected = pd.MultiIndex.from_product([['a', 'b', 'c'], [1, 2]], names=['y', 'x'])
    actual = da.coords.to_index(['y', 'x'])
    assert expected.equals(actual)
    with pytest.raises(ValueError, match='ordered_dims must match'):
        da.coords.to_index(['x'])