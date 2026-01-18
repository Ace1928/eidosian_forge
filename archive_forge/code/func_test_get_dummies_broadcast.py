from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_get_dummies_broadcast(dtype) -> None:
    values = xr.DataArray(['x~x|x~x', 'x', 'x|x~x', 'x~x'], dims=['X']).astype(dtype)
    sep = xr.DataArray(['|', '~'], dims=['Y']).astype(dtype)
    expected_list = [[[False, False, True], [True, True, False]], [[True, False, False], [True, False, False]], [[True, False, True], [True, True, False]], [[False, False, True], [True, False, False]]]
    expected_np = np.array(expected_list)
    expected = xr.DataArray(expected_np, dims=['X', 'Y', 'ZZ'])
    expected.coords['ZZ'] = np.array(['x', 'x|x', 'x~x']).astype(dtype)
    res = values.str.get_dummies(dim='ZZ', sep=sep)
    assert res.dtype == expected.dtype
    assert_equal(res, expected)