from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_join_vector(dtype) -> None:
    values = xr.DataArray(['a', 'bb', 'cccc'], dims=['Y']).astype(dtype)
    targ_blank = xr.DataArray('abbcccc').astype(dtype)
    targ_space = xr.DataArray('a bb cccc').astype(dtype)
    res_blank_none = values.str.join()
    res_blank_y = values.str.join(dim='Y')
    res_space_none = values.str.join(sep=' ')
    res_space_y = values.str.join(dim='Y', sep=' ')
    assert res_blank_none.dtype == targ_blank.dtype
    assert res_blank_y.dtype == targ_blank.dtype
    assert res_space_none.dtype == targ_space.dtype
    assert res_space_y.dtype == targ_space.dtype
    assert_identical(res_blank_none, targ_blank)
    assert_identical(res_blank_y, targ_blank)
    assert_identical(res_space_none, targ_space)
    assert_identical(res_space_y, targ_space)