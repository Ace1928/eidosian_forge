from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_get_dummies_empty(dtype) -> None:
    values = xr.DataArray([], dims=['X']).astype(dtype)
    expected = xr.DataArray(np.zeros((0, 0)), dims=['X', 'ZZ']).astype(dtype)
    res = values.str.get_dummies(dim='ZZ')
    assert res.dtype == expected.dtype
    assert_equal(res, expected)