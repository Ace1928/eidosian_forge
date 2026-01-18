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
def test_groupby_none_group_name() -> None:
    data = np.arange(10) + 10
    da = xr.DataArray(data)
    key = xr.DataArray(np.floor_divide(data, 2))
    mean = da.groupby(key).mean()
    assert 'group' in mean.dims