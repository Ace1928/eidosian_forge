from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
@requires_dask
def test_corr_dtype_error():
    da_a = xr.DataArray([[1, 2], [2, 1]], dims=['x', 'time'])
    da_b = xr.DataArray([[1, 2], [1, np.nan]], dims=['x', 'time'])
    xr.testing.assert_equal(xr.corr(da_a, da_b), xr.corr(da_a.chunk(), da_b.chunk()))
    xr.testing.assert_equal(xr.corr(da_a, da_b), xr.corr(da_a, da_b.chunk()))