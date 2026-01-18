from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
@pytest.mark.parametrize('dim', ['x', 'y'])
@pytest.mark.parametrize('window', [3, 8, 11])
@pytest.mark.parametrize('center', [True, False])
def test_dask_rolling(self, dim, window, center):
    import dask
    import dask.array as da
    dask.config.set(scheduler='single-threaded')
    x = Variable(('x', 'y'), np.array(np.random.randn(100, 40), dtype=float))
    dx = Variable(('x', 'y'), da.from_array(x, chunks=[(6, 30, 30, 20, 14), 8]))
    expected = x.rolling_window(dim, window, 'window', center=center, fill_value=np.nan)
    with raise_if_dask_computes():
        actual = dx.rolling_window(dim, window, 'window', center=center, fill_value=np.nan)
    assert isinstance(actual.data, da.Array)
    assert actual.shape == expected.shape
    assert_equal(actual, expected)