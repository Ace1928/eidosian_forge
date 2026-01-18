from __future__ import annotations
import numpy as np
from xarray import DataArray
from datashader.datashape import dshape
from datashader.utils import Dispatcher, apply, calc_res, isreal, orient_array
def test_isreal():
    assert isreal('int32')
    assert isreal(dshape('int32'))
    assert isreal('?int32')
    assert isreal('float64')
    assert not isreal('complex64')
    assert not isreal('{x: int64, y: float64}')