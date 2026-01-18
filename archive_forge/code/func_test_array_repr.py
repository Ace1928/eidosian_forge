from __future__ import annotations
import sys
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray
import xarray as xr
from xarray.core import formatting
from xarray.tests import requires_cftime, requires_dask, requires_netCDF4
def test_array_repr(self) -> None:
    ds = xr.Dataset(coords={'foo': np.array([1, 2, 3], dtype=np.uint64), 'bar': np.array([1, 2, 3], dtype=np.uint64)})
    ds[1, 2] = xr.DataArray(np.array([0], dtype=np.uint64), dims='test')
    ds_12 = ds[1, 2]
    actual = formatting.array_repr(ds_12)
    expected = dedent('        <xarray.DataArray (1, 2) (test: 1)> Size: 8B\n        array([0], dtype=uint64)\n        Dimensions without coordinates: test')
    assert actual == expected
    assert repr(ds_12) == expected
    assert str(ds_12) == expected
    actual = f'{ds_12}'
    assert actual == expected
    with xr.set_options(display_expand_data=False):
        actual = formatting.array_repr(ds[1, 2])
        expected = dedent('            <xarray.DataArray (1, 2) (test: 1)> Size: 8B\n            0\n            Dimensions without coordinates: test')
        assert actual == expected