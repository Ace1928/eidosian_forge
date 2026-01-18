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
@pytest.mark.parametrize('ddof', [0, 1])
@pytest.mark.parametrize('n', [3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize('dim', [None, 'x', 'time'])
@requires_dask
def test_lazy_corrcov(n: int, dim: str | None, ddof: int, array_tuples: tuple[xr.DataArray, xr.DataArray]) -> None:
    from dask import is_dask_collection
    da_a, da_b = array_tuples[n]
    with raise_if_dask_computes():
        cov = xr.cov(da_a.chunk(), da_b.chunk(), dim=dim, ddof=ddof)
        assert is_dask_collection(cov)
        corr = xr.corr(da_a.chunk(), da_b.chunk(), dim=dim)
        assert is_dask_collection(corr)