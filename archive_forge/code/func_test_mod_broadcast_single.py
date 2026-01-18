from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_mod_broadcast_single() -> None:
    values = xr.DataArray(['%s_1', '%s_2', '%s_3'], dims=['X']).astype(np.str_)
    pos = xr.DataArray(['2.3', '3.44444'], dims=['YY'])
    expected = xr.DataArray([['2.3_1', '3.44444_1'], ['2.3_2', '3.44444_2'], ['2.3_3', '3.44444_3']], dims=['X', 'YY']).astype(np.str_)
    res = values.str % pos
    assert res.dtype == expected.dtype
    assert_equal(res, expected)