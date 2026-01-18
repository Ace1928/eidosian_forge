from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
def test_ffill_use_bottleneck_numbagg():
    da = xr.DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims='x')
    with xr.set_options(use_bottleneck=False, use_numbagg=False):
        with pytest.raises(RuntimeError):
            da.ffill('x')