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
def test_constructor_invalid(self) -> None:
    data = np.random.randn(3, 2)
    with pytest.raises(ValueError, match='coords is not dict-like'):
        DataArray(data, [[0, 1, 2]], ['x', 'y'])
    with pytest.raises(ValueError, match='not a subset of the .* dim'):
        DataArray(data, {'x': [0, 1, 2]}, ['a', 'b'])
    with pytest.raises(ValueError, match='not a subset of the .* dim'):
        DataArray(data, {'x': [0, 1, 2]})
    with pytest.raises(TypeError, match='is not hashable'):
        DataArray(data, dims=['x', []])
    with pytest.raises(ValueError, match='conflicting sizes for dim'):
        DataArray([1, 2, 3], coords=[('x', [0, 1])])
    with pytest.raises(ValueError, match='conflicting sizes for dim'):
        DataArray([1, 2], coords={'x': [0, 1], 'y': ('x', [1])}, dims='x')
    with pytest.raises(ValueError, match='conflicting MultiIndex'):
        DataArray(np.random.rand(4, 4), [('x', self.mindex), ('y', self.mindex)])
    with pytest.raises(ValueError, match='conflicting MultiIndex'):
        DataArray(np.random.rand(4, 4), [('x', self.mindex), ('level_1', range(4))])