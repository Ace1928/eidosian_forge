from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
def test_groupby_bins_ellipsis(self) -> None:
    da = xr.DataArray(np.ones((2, 3, 4)))
    bins = [-1, 0, 1, 2]
    with xr.set_options(use_flox=False):
        actual = da.groupby_bins('dim_0', bins).mean(...)
    with xr.set_options(use_flox=True):
        expected = da.groupby_bins('dim_0', bins).mean(...)
    assert_allclose(actual, expected)