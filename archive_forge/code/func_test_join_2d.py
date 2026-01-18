from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_join_2d(dtype) -> None:
    values = xr.DataArray([['a', 'bb', 'cccc'], ['ddddd', 'eeee', 'fff']], dims=['X', 'Y']).astype(dtype)
    targ_blank_x = xr.DataArray(['addddd', 'bbeeee', 'ccccfff'], dims=['Y']).astype(dtype)
    targ_space_x = xr.DataArray(['a ddddd', 'bb eeee', 'cccc fff'], dims=['Y']).astype(dtype)
    targ_blank_y = xr.DataArray(['abbcccc', 'dddddeeeefff'], dims=['X']).astype(dtype)
    targ_space_y = xr.DataArray(['a bb cccc', 'ddddd eeee fff'], dims=['X']).astype(dtype)
    res_blank_x = values.str.join(dim='X')
    res_blank_y = values.str.join(dim='Y')
    res_space_x = values.str.join(dim='X', sep=' ')
    res_space_y = values.str.join(dim='Y', sep=' ')
    assert res_blank_x.dtype == targ_blank_x.dtype
    assert res_blank_y.dtype == targ_blank_y.dtype
    assert res_space_x.dtype == targ_space_x.dtype
    assert res_space_y.dtype == targ_space_y.dtype
    assert_identical(res_blank_x, targ_blank_x)
    assert_identical(res_blank_y, targ_blank_y)
    assert_identical(res_space_x, targ_space_x)
    assert_identical(res_space_y, targ_space_y)
    with pytest.raises(ValueError, match='Dimension must be specified for multidimensional arrays.'):
        values.str.join()