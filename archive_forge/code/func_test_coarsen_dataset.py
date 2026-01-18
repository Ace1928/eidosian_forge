from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.core import duck_array_ops
from xarray.tests import (
@pytest.mark.parametrize('dask', [True, False])
@pytest.mark.parametrize(('boundary', 'side'), [('trim', 'left'), ('pad', 'right')])
def test_coarsen_dataset(ds, dask, boundary, side):
    if dask and has_dask:
        ds = ds.chunk({'x': 4})
    actual = ds.coarsen(time=2, x=3, boundary=boundary, side=side).max()
    assert_equal(actual['z1'], ds['z1'].coarsen(x=3, boundary=boundary, side=side).max())
    assert_equal(actual['time'], ds['time'].coarsen(time=2, boundary=boundary, side=side).mean())