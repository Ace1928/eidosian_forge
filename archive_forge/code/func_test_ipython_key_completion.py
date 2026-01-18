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
def test_ipython_key_completion(self) -> None:
    ds = create_test_data(seed=1)
    actual = ds._ipython_key_completions_()
    expected = ['var1', 'var2', 'var3', 'time', 'dim1', 'dim2', 'dim3', 'numbers']
    for item in actual:
        ds[item]
    assert sorted(actual) == sorted(expected)
    actual = ds['var3']._ipython_key_completions_()
    expected = ['dim3', 'dim1', 'numbers']
    for item in actual:
        ds['var3'][item]
    assert sorted(actual) == sorted(expected)
    ds_midx = ds.stack(dim12=['dim2', 'dim3'])
    actual = ds_midx._ipython_key_completions_()
    expected = ['var1', 'var2', 'var3', 'time', 'dim1', 'dim2', 'dim3', 'numbers', 'dim12']
    for item in actual:
        ds_midx[item]
    assert sorted(actual) == sorted(expected)
    actual = ds.coords._ipython_key_completions_()
    expected = ['time', 'dim1', 'dim2', 'dim3', 'numbers']
    for item in actual:
        ds.coords[item]
    assert sorted(actual) == sorted(expected)
    actual = ds['var3'].coords._ipython_key_completions_()
    expected = ['dim1', 'dim3', 'numbers']
    for item in actual:
        ds['var3'].coords[item]
    assert sorted(actual) == sorted(expected)
    coords = Coordinates(ds.coords)
    actual = coords._ipython_key_completions_()
    expected = ['time', 'dim2', 'dim3', 'numbers']
    for item in actual:
        coords[item]
    assert sorted(actual) == sorted(expected)
    actual = ds.data_vars._ipython_key_completions_()
    expected = ['var1', 'var2', 'var3', 'dim1']
    for item in actual:
        ds.data_vars[item]
    assert sorted(actual) == sorted(expected)