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
@pytest.mark.parametrize('use_flox', [True, False])
def test_groupby_dim_no_dim_equal(use_flox: bool) -> None:
    da = DataArray(data=[1, 2, 3, 4], dims='lat', coords={'lat': np.linspace(0, 1.01, 4)})
    with xr.set_options(use_flox=use_flox):
        actual1 = da.drop_vars('lat').groupby('lat', squeeze=False).sum()
        actual2 = da.groupby('lat', squeeze=False).sum()
    assert_identical(actual1, actual2.drop_vars('lat'))