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
def test_interpolate_dask():
    da, _ = make_interpolate_example_data((40, 40), 0.5)
    da = da.chunk({'x': 5})
    actual = da.interpolate_na('time')
    expected = da.load().interpolate_na('time')
    assert isinstance(actual.data, dask_array_type)
    assert_equal(actual.compute(), expected)
    da = da.chunk({'x': 5})
    actual = da.interpolate_na('time', limit=3)
    expected = da.load().interpolate_na('time', limit=3)
    assert isinstance(actual.data, dask_array_type)
    assert_equal(actual, expected)