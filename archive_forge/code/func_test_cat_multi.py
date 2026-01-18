from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_cat_multi() -> None:
    values_1 = xr.DataArray(['11111', '4'], dims=['X'])
    values_2 = xr.DataArray(['a', 'bb', 'cccc'], dims=['Y']).astype(np.bytes_)
    values_3 = np.array(3.4)
    values_4 = ''
    values_5 = np.array('', dtype=np.str_)
    sep = xr.DataArray([' ', ', '], dims=['ZZ']).astype(np.str_)
    expected = xr.DataArray([[['11111 a 3.4  ', '11111, a, 3.4, , '], ['11111 bb 3.4  ', '11111, bb, 3.4, , '], ['11111 cccc 3.4  ', '11111, cccc, 3.4, , ']], [['4 a 3.4  ', '4, a, 3.4, , '], ['4 bb 3.4  ', '4, bb, 3.4, , '], ['4 cccc 3.4  ', '4, cccc, 3.4, , ']]], dims=['X', 'Y', 'ZZ']).astype(np.str_)
    res = values_1.str.cat(values_2, values_3, values_4, values_5, sep=sep)
    assert res.dtype == expected.dtype
    assert_equal(res, expected)