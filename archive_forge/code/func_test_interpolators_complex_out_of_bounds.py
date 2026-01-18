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
def test_interpolators_complex_out_of_bounds():
    """Ensure complex nans are used for complex data"""
    xi = np.array([-1, 0, 1, 2, 5], dtype=np.float64)
    yi = np.exp(1j * xi)
    x = np.array([-2, 1, 6], dtype=np.float64)
    expected = np.array([np.nan + np.nan * 1j, np.exp(1j), np.nan + np.nan * 1j], dtype=yi.dtype)
    for method, interpolator in [('linear', NumpyInterpolator), ('linear', ScipyInterpolator)]:
        f = interpolator(xi, yi, method=method)
        actual = f(x)
        assert_array_equal(actual, expected)