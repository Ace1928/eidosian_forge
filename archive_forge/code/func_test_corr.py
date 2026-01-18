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
@pytest.mark.parametrize('n', [0, 1, 2])
@pytest.mark.parametrize('dim', [None, 'time'])
def test_corr(n: int, dim: str | None, array_tuples: tuple[xr.DataArray, xr.DataArray]) -> None:
    da_a, da_b = array_tuples[n]
    if dim is not None:

        def np_corr_ind(ts1, ts2, a, x):
            ts1, ts2 = broadcast(ts1, ts2)
            valid_values = ts1.notnull() & ts2.notnull()
            ts1 = ts1.where(valid_values)
            ts2 = ts2.where(valid_values)
            return np.ma.corrcoef(np.ma.masked_invalid(ts1.sel(a=a, x=x).data.flatten()), np.ma.masked_invalid(ts2.sel(a=a, x=x).data.flatten()))[0, 1]
        expected = np.zeros((3, 4))
        for a in [0, 1, 2]:
            for x in [0, 1, 2, 3]:
                expected[a, x] = np_corr_ind(da_a, da_b, a=a, x=x)
        actual = xr.corr(da_a, da_b, dim)
        assert_allclose(actual, expected)
    else:

        def np_corr(ts1, ts2):
            ts1, ts2 = broadcast(ts1, ts2)
            valid_values = ts1.notnull() & ts2.notnull()
            ts1 = ts1.where(valid_values)
            ts2 = ts2.where(valid_values)
            return np.ma.corrcoef(np.ma.masked_invalid(ts1.data.flatten()), np.ma.masked_invalid(ts2.data.flatten()))[0, 1]
        expected = np_corr(da_a, da_b)
        actual = xr.corr(da_a, da_b, dim)
        assert_allclose(actual, expected)