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
def test_groupby_dataset_order() -> None:
    ds = Dataset()
    for vn in ['a', 'b', 'c']:
        ds[vn] = DataArray(np.arange(10), dims=['t'])
    data_vars_ref = list(ds.data_vars.keys())
    ds = ds.groupby('t').mean(...)
    data_vars = list(ds.data_vars.keys())
    assert data_vars == data_vars_ref