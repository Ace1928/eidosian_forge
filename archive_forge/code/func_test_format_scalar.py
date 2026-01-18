from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_format_scalar() -> None:
    values = xr.DataArray(['{}.{Y}.{ZZ}', '{},{},{X},{X}', '{X}-{Y}-{ZZ}'], dims=['X']).astype(np.str_)
    pos0 = 1
    pos1 = 1.2
    pos2 = '2.3'
    X = "'test'"
    Y = 'X'
    ZZ = None
    W = 'NO!'
    expected = xr.DataArray(['1.X.None', "1,1.2,'test','test'", "'test'-X-None"], dims=['X']).astype(np.str_)
    res = values.str.format(pos0, pos1, pos2, X=X, Y=Y, ZZ=ZZ, W=W)
    assert res.dtype == expected.dtype
    assert_equal(res, expected)