from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_join_scalar(dtype) -> None:
    values = xr.DataArray('aaa').astype(dtype)
    targ = xr.DataArray('aaa').astype(dtype)
    res_blank = values.str.join()
    res_space = values.str.join(sep=' ')
    assert res_blank.dtype == targ.dtype
    assert res_space.dtype == targ.dtype
    assert_identical(res_blank, targ)
    assert_identical(res_space, targ)