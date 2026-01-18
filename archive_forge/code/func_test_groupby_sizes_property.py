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
def test_groupby_sizes_property(dataset) -> None:
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        assert dataset.groupby('x').sizes == dataset.isel(x=1).sizes
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        assert dataset.groupby('y').sizes == dataset.isel(y=1).sizes
    stacked = dataset.stack({'xy': ('x', 'y')})
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        assert stacked.groupby('xy').sizes == stacked.isel(xy=0).sizes