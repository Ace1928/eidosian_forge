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
@pytest.mark.parametrize('n', range(9))
@pytest.mark.parametrize('dim', [None, 'time', 'x'])
@pytest.mark.filterwarnings('ignore:invalid value encountered in .*divide')
def test_corr_lazycorr_consistency(n: int, dim: str | None, array_tuples: tuple[xr.DataArray, xr.DataArray]) -> None:
    da_a, da_b = array_tuples[n]
    da_al = da_a.chunk()
    da_bl = da_b.chunk()
    c_abl = xr.corr(da_al, da_bl, dim=dim)
    c_ab = xr.corr(da_a, da_b, dim=dim)
    c_ab_mixed = xr.corr(da_a, da_bl, dim=dim)
    assert_allclose(c_ab, c_abl)
    assert_allclose(c_ab, c_ab_mixed)