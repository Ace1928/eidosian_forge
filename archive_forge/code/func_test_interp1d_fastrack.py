from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_scipy
@pytest.mark.parametrize('method,vals', [pytest.param(method, vals, id=f'{desc}:{method}') for method in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial'] for desc, vals in [('no nans', np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)), ('one nan', np.array([1, np.nan, np.nan], dtype=np.float64)), ('all nans', np.full(6, np.nan, dtype=np.float64))]])
def test_interp1d_fastrack(method, vals):
    expected = xr.DataArray(vals, dims='x')
    actual = expected.interpolate_na(dim='x', method=method)
    assert_equal(actual, expected)