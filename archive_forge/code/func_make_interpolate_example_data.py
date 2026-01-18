from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
def make_interpolate_example_data(shape, frac_nan, seed=12345, non_uniform=False):
    rs = np.random.RandomState(seed)
    vals = rs.normal(size=shape)
    if frac_nan == 1:
        vals[:] = np.nan
    elif frac_nan == 0:
        pass
    else:
        n_missing = int(vals.size * frac_nan)
        ys = np.arange(shape[0])
        xs = np.arange(shape[1])
        if n_missing:
            np.random.shuffle(ys)
            ys = ys[:n_missing]
            np.random.shuffle(xs)
            xs = xs[:n_missing]
            vals[ys, xs] = np.nan
    if non_uniform:
        deltas = pd.TimedeltaIndex(unit='d', data=rs.normal(size=shape[0], scale=10))
        coords = {'time': (pd.Timestamp('2000-01-01') + deltas).sort_values()}
    else:
        coords = {'time': pd.date_range('2000-01-01', freq='D', periods=shape[0])}
    da = xr.DataArray(vals, dims=('time', 'x'), coords=coords)
    df = da.to_pandas()
    return (da, df)