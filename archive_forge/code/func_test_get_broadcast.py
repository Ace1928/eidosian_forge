from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_get_broadcast(dtype) -> None:
    values = xr.DataArray(['a_b_c', 'c_d_e', 'f_g_h'], dims=['X']).astype(dtype)
    inds = xr.DataArray([0, 2], dims=['Y'])
    result = values.str.get(inds)
    expected = xr.DataArray([['a', 'b'], ['c', 'd'], ['f', 'g']], dims=['X', 'Y']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)