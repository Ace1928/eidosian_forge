from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_join_broadcast(dtype) -> None:
    values = xr.DataArray(['a', 'bb', 'cccc'], dims=['X']).astype(dtype)
    sep = xr.DataArray([' ', ', '], dims=['ZZ']).astype(dtype)
    expected = xr.DataArray(['a bb cccc', 'a, bb, cccc'], dims=['ZZ']).astype(dtype)
    res = values.str.join(sep=sep)
    assert res.dtype == expected.dtype
    assert_identical(res, expected)