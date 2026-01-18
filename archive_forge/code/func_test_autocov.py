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
@pytest.mark.parametrize('n', range(5))
@pytest.mark.parametrize('dim', [None, 'time', 'x', ['time', 'x']])
def test_autocov(n: int, dim: str | None, arrays) -> None:
    da = arrays[n]
    valid_values = da.notnull()
    da = da.where(valid_values.sum(dim=dim) > 1)
    expected = ((da - da.mean(dim=dim)) ** 2).sum(dim=dim, skipna=True, min_count=1)
    actual = xr.cov(da, da, dim=dim) * (valid_values.sum(dim) - 1)
    assert_allclose(actual, expected)