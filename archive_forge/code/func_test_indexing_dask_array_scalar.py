from __future__ import annotations
import itertools
from typing import Any
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable
from xarray.core import indexing, nputils
from xarray.core.indexes import PandasIndex, PandasMultiIndex
from xarray.core.types import T_Xarray
from xarray.tests import (
@requires_dask
def test_indexing_dask_array_scalar():
    import dask.array
    a = dask.array.from_array(np.linspace(0.0, 1.0))
    da = DataArray(a, dims='x')
    x_selector = da.argmax(dim=...)
    with raise_if_dask_computes():
        actual = da.isel(x_selector)
    expected = da.isel(x=-1)
    assert_identical(actual, expected)