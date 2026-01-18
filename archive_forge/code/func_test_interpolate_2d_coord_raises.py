from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
def test_interpolate_2d_coord_raises():
    coords = {'x': xr.Variable(('a', 'b'), np.arange(6).reshape(2, 3)), 'y': xr.Variable(('a', 'b'), np.arange(6).reshape(2, 3)) * 2}
    data = np.random.randn(2, 3)
    data[1, 1] = np.nan
    da = xr.DataArray(data, dims=('a', 'b'), coords=coords)
    with pytest.raises(ValueError, match='interpolation must be 1D'):
        da.interpolate_na(dim='a', use_coordinate='x')