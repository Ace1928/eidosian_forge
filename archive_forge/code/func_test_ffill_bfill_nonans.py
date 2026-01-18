from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_bottleneck
def test_ffill_bfill_nonans():
    vals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    expected = xr.DataArray(vals, dims='x')
    actual = expected.ffill(dim='x')
    assert_equal(actual, expected)
    actual = expected.bfill(dim='x')
    assert_equal(actual, expected)