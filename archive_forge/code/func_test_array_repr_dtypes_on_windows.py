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
@pytest.mark.skipif(not ON_WINDOWS, reason="Default numpy's dtypes vary according to OS")
def test_array_repr_dtypes_on_windows() -> None:
    ds = xr.DataArray(np.array([0]), dims='x')
    actual = repr(ds)
    expected = '\n<xarray.DataArray (x: 1)> Size: 4B\narray([0])\nDimensions without coordinates: x\n        '.strip()
    assert actual == expected
    ds = xr.DataArray(np.array([0], dtype='int32'), dims='x')
    actual = repr(ds)
    expected = '\n<xarray.DataArray (x: 1)> Size: 4B\narray([0])\nDimensions without coordinates: x\n        '.strip()
    assert actual == expected
    ds = xr.DataArray(np.array([0], dtype='int64'), dims='x')
    actual = repr(ds)
    expected = '\n<xarray.DataArray (x: 1)> Size: 8B\narray([0], dtype=int64)\nDimensions without coordinates: x\n        '.strip()
    assert actual == expected