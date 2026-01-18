from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_mod_scalar() -> None:
    values = xr.DataArray(['%s.%s.%s', '%s,%s,%s', '%s-%s-%s'], dims=['X']).astype(np.str_)
    pos0 = 1
    pos1 = 1.2
    pos2 = '2.3'
    expected = xr.DataArray(['1.1.2.2.3', '1,1.2,2.3', '1-1.2-2.3'], dims=['X']).astype(np.str_)
    res = values.str % (pos0, pos1, pos2)
    assert res.dtype == expected.dtype
    assert_equal(res, expected)