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
def test_loc_assign(self) -> None:
    self.ds['x'] = ('x', np.array(list('abcdefghij')))
    da = self.ds['foo']
    da.loc['a':'j'] = 0
    assert np.all(da.values == 0)
    da.loc[{'x': slice('a', 'j')}] = 2
    assert np.all(da.values == 2)
    da.loc[{'x': slice('a', 'j')}] = 2
    assert np.all(da.values == 2)
    da = DataArray(np.arange(12).reshape(3, 4), dims=['x', 'y'])
    da.loc[0, 0] = 0
    assert da.values[0, 0] == 0
    assert da.values[0, 1] != 0
    da = DataArray(np.arange(12).reshape(3, 4), dims=['x', 'y'])
    da.loc[0] = 0
    assert np.all(da.values[0] == np.zeros(4))
    assert da.values[1, 0] != 0