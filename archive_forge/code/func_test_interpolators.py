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
def test_interpolators():
    for method, interpolator in [('linear', NumpyInterpolator), ('linear', ScipyInterpolator), ('spline', SplineInterpolator)]:
        xi = np.array([-1, 0, 1, 2, 5], dtype=np.float64)
        yi = np.array([-10, 0, 10, 20, 50], dtype=np.float64)
        x = np.array([3, 4], dtype=np.float64)
        f = interpolator(xi, yi, method=method)
        out = f(x)
        assert pd.isnull(out).sum() == 0