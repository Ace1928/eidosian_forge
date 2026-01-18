from __future__ import annotations
import numpy as np
from xarray import DataArray
from datashader.datashape import dshape
from datashader.utils import Dispatcher, apply, calc_res, isreal, orient_array
def test_calc_res():
    x = [5, 7]
    y = [0, 1]
    z = [[0, 0], [0, 0]]
    dims = ('y', 'x')
    xarr = DataArray(z, coords=dict(x=x, y=y), dims=dims)
    xres, yres = calc_res(xarr)
    assert xres == 2
    assert yres == -1
    xarr = DataArray(z, coords=dict(x=x, y=y[::-1]), dims=dims)
    xres, yres = calc_res(xarr)
    assert xres == 2
    assert yres == 1
    xarr = DataArray(z, coords=dict(x=x[::-1], y=y), dims=dims)
    xres, yres = calc_res(xarr)
    assert xres == -2
    assert yres == -1
    xarr = DataArray(z, coords=dict(x=x[::-1], y=y[::-1]), dims=dims)
    xres, yres = calc_res(xarr)
    assert xres == -2
    assert yres == 1