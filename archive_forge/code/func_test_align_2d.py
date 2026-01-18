from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
@pytest.mark.xfail(reason='COO objects currently do not accept more than one iterable index at a time')
def test_align_2d(self):
    A1 = xr.DataArray(self.sp_ar, dims=['x', 'y'], coords={'x': np.arange(self.sp_ar.shape[0]), 'y': np.arange(self.sp_ar.shape[1])})
    A2 = xr.DataArray(self.sp_ar, dims=['x', 'y'], coords={'x': np.arange(1, self.sp_ar.shape[0] + 1), 'y': np.arange(1, self.sp_ar.shape[1] + 1)})
    B1, B2 = xr.align(A1, A2, join='inner')
    assert np.all(B1.coords['x'] == np.arange(1, self.sp_ar.shape[0]))
    assert np.all(B1.coords['y'] == np.arange(1, self.sp_ar.shape[0]))
    assert np.all(B1.coords['x'] == B2.coords['x'])
    assert np.all(B1.coords['y'] == B2.coords['y'])