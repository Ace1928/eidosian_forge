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
@requires_dask
@pytest.mark.parametrize('method', ['ffill', 'bfill'])
def test_ffill_bfill_dask(method):
    da, _ = make_interpolate_example_data((40, 40), 0.5)
    da = da.chunk({'x': 5})
    dask_method = getattr(da, method)
    numpy_method = getattr(da.compute(), method)
    with raise_if_dask_computes():
        actual = dask_method('time')
    expected = numpy_method('time')
    assert_equal(actual, expected)
    with raise_if_dask_computes():
        actual = dask_method('x')
    expected = numpy_method('x')
    assert_equal(actual, expected)
    with raise_if_dask_computes():
        actual = dask_method('time', limit=3)
    expected = numpy_method('time', limit=3)
    assert_equal(actual, expected)
    with raise_if_dask_computes():
        actual = dask_method('x', limit=2)
    expected = numpy_method('x', limit=2)
    assert_equal(actual, expected)
    with raise_if_dask_computes():
        actual = dask_method('x', limit=41)
    expected = numpy_method('x', limit=41)
    assert_equal(actual, expected)