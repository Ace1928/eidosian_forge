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
def test_interpolate_pd_compat_non_uniform_index():
    shapes = [(8, 8), (1, 20), (20, 1), (100, 100)]
    frac_nans = [0, 0.5, 1]
    methods = ['time', 'index', 'values']
    for shape, frac_nan, method in itertools.product(shapes, frac_nans, methods):
        da, df = make_interpolate_example_data(shape, frac_nan, non_uniform=True)
        for dim in ['time', 'x']:
            if method == 'time' and dim != 'time':
                continue
            actual = da.interpolate_na(method='linear', dim=dim, use_coordinate=True, fill_value=np.nan)
            expected = df.interpolate(method=method, axis=da.get_axis_num(dim))
            expected.values[pd.isnull(actual.values)] = np.nan
            np.testing.assert_allclose(actual.values, expected.values)