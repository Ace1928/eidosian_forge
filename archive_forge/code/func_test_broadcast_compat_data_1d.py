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
def test_broadcast_compat_data_1d() -> None:
    data = np.arange(5)
    var = xr.Variable('x', data)
    assert_identical(data, broadcast_compat_data(var, ('x',), ()))
    assert_identical(data, broadcast_compat_data(var, (), ('x',)))
    assert_identical(data[:], broadcast_compat_data(var, ('w',), ('x',)))
    assert_identical(data[:, None], broadcast_compat_data(var, ('w', 'x', 'y'), ()))
    with pytest.raises(ValueError):
        broadcast_compat_data(var, ('x',), ('w',))
    with pytest.raises(ValueError):
        broadcast_compat_data(var, (), ())