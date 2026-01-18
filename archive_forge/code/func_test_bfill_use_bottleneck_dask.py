from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_dask
def test_bfill_use_bottleneck_dask():
    da = xr.DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims='x')
    da = da.chunk({'x': 1})
    with xr.set_options(use_bottleneck=False, use_numbagg=False):
        with pytest.raises(RuntimeError):
            da.bfill('x')