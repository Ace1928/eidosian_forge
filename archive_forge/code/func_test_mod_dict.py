from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_mod_dict() -> None:
    values = xr.DataArray(['%(a)s.%(a)s.%(b)s', '%(b)s,%(c)s,%(b)s', '%(c)s-%(b)s-%(a)s'], dims=['X']).astype(np.str_)
    a = 1
    b = 1.2
    c = '2.3'
    expected = xr.DataArray(['1.1.1.2', '1.2,2.3,1.2', '2.3-1.2-1'], dims=['X']).astype(np.str_)
    res = values.str % {'a': a, 'b': b, 'c': c}
    assert res.dtype == expected.dtype
    assert_equal(res, expected)