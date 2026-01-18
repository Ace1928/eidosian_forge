from __future__ import annotations
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy import array, nan
from xarray import DataArray, Dataset, cftime_range, concat
from xarray.core import dtypes, duck_array_ops
from xarray.core.duck_array_ops import (
from xarray.namedarray.pycompat import array_type
from xarray.testing import assert_allclose, assert_equal, assert_identical
from xarray.tests import (
def test_cumsum_1d():
    inputs = np.array([0, 1, 2, 3])
    expected = np.array([0, 1, 3, 6])
    actual = duck_array_ops.cumsum(inputs)
    assert_array_equal(expected, actual)
    actual = duck_array_ops.cumsum(inputs, axis=0)
    assert_array_equal(expected, actual)
    actual = duck_array_ops.cumsum(inputs, axis=-1)
    assert_array_equal(expected, actual)
    actual = duck_array_ops.cumsum(inputs, axis=(0,))
    assert_array_equal(expected, actual)
    actual = duck_array_ops.cumsum(inputs, axis=())
    assert_array_equal(inputs, actual)