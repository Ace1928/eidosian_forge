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
@pytest.mark.parametrize('method', ['load', 'compute', 'persist'])
def test_dask_kwargs_dataarray(method):
    data = da.from_array(np.arange(3), chunks=(2,))
    x = DataArray(data)
    if method in ['load', 'compute']:
        dask_func = 'dask.array.compute'
    else:
        dask_func = 'dask.persist'
    with mock.patch(dask_func) as mock_func:
        getattr(x, method)(foo='bar')
    mock_func.assert_called_with(data, foo='bar')