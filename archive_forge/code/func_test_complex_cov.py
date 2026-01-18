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
def test_complex_cov() -> None:
    da = xr.DataArray([1j, -1j])
    actual = xr.cov(da, da)
    assert abs(actual.item()) == 2