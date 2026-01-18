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
def test_advanced_indexing_dask_array():
    import dask.array as da
    ds = Dataset(dict(a=('x', da.from_array(np.random.randint(0, 100, 100))), b=(('x', 'y'), da.random.random((100, 10)))))
    expected = ds.b.sel(x=ds.a.compute())
    with raise_if_dask_computes():
        actual = ds.b.sel(x=ds.a)
    assert_identical(expected, actual)
    with raise_if_dask_computes():
        actual = ds.b.sel(x=ds.a.data)
    assert_identical(expected, actual)