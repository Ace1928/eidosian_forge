from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize('deep, expected_orig', [[True, xr.DataArray(xr.IndexVariable('a', np.array([1, 2])), coords={'a': [1, 2]}, dims=['a'])], [False, xr.DataArray(xr.IndexVariable('a', np.array([999, 2])), coords={'a': [999, 2]}, dims=['a'])]])
def test_copy_coords(self, deep, expected_orig) -> None:
    """The test fails for the shallow copy, and apparently only on Windows
        for some reason. In windows coords seem to be immutable unless it's one
        dataset deep copied from another."""
    ds = xr.DataArray(np.ones([2, 2, 2]), coords={'a': [1, 2], 'b': ['x', 'y'], 'c': [0, 1]}, dims=['a', 'b', 'c'], name='value').to_dataset()
    ds_cp = ds.copy(deep=deep)
    new_a = np.array([999, 2])
    ds_cp.coords['a'] = ds_cp.a.copy(data=new_a)
    expected_cp = xr.DataArray(xr.IndexVariable('a', new_a), coords={'a': [999, 2]}, dims=['a'])
    assert_identical(ds_cp.coords['a'], expected_cp)
    assert_identical(ds.coords['a'], expected_orig)