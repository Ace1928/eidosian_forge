from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
def test_dataarray_with_dask_coords():
    import toolz
    x = xr.Variable('x', da.arange(8, chunks=(4,)))
    y = xr.Variable('y', da.arange(8, chunks=(4,)) * 2)
    data = da.random.random((8, 8), chunks=(4, 4)) + 1
    array = xr.DataArray(data, dims=['x', 'y'])
    array.coords['xx'] = x
    array.coords['yy'] = y
    assert dict(array.__dask_graph__()) == toolz.merge(data.__dask_graph__(), x.__dask_graph__(), y.__dask_graph__())
    array2, = dask.compute(array)
    assert not dask.is_dask_collection(array2)
    assert all((isinstance(v._variable.data, np.ndarray) for v in array2.coords.values()))