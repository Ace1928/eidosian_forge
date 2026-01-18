from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
@requires_dask
def test_apply_dask_parallelized_two_outputs() -> None:
    data_array = xr.DataArray([[0, 1, 2], [1, 2, 3]], dims=('x', 'y'))

    def twice(obj):

        def func(x):
            return (x, x)
        return apply_ufunc(func, obj, output_core_dims=[[], []], dask='parallelized')
    out0, out1 = twice(data_array.chunk({'x': 1}))
    assert_identical(data_array, out0)
    assert_identical(data_array, out1)