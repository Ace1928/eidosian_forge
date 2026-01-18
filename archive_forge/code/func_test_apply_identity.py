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
def test_apply_identity() -> None:
    array = np.arange(10)
    variable = xr.Variable('x', array)
    data_array = xr.DataArray(variable, [('x', -array)])
    dataset = xr.Dataset({'y': variable}, {'x': -array})
    apply_identity = functools.partial(apply_ufunc, identity)
    assert_identical(array, apply_identity(array))
    assert_identical(variable, apply_identity(variable))
    assert_identical(data_array, apply_identity(data_array))
    assert_identical(data_array, apply_identity(data_array.groupby('x')))
    assert_identical(data_array, apply_identity(data_array.groupby('x', squeeze=False)))
    assert_identical(dataset, apply_identity(dataset))
    assert_identical(dataset, apply_identity(dataset.groupby('x')))
    assert_identical(dataset, apply_identity(dataset.groupby('x', squeeze=False)))