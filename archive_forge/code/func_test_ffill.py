from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_numbagg_or_bottleneck
def test_ffill():
    da = xr.DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims='x')
    expected = xr.DataArray(np.array([4, 5, 5], dtype=np.float64), dims='x')
    actual = da.ffill('x')
    assert_equal(actual, expected)