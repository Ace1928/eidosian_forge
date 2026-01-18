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
@pytest.mark.filterwarnings('ignore:return type')
def test_groupby_dims_property(dataset, recwarn) -> None:
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        assert dataset.groupby('x').dims == dataset.isel(x=1).dims
        assert dataset.groupby('y').dims == dataset.isel(y=1).dims
    recwarn.clear()
    assert tuple(dataset.groupby('x', squeeze=False).dims) == tuple(dataset.isel(x=slice(1, 2)).dims)
    assert tuple(dataset.groupby('y', squeeze=False).dims) == tuple(dataset.isel(y=slice(1, 2)).dims)
    assert len(recwarn) == 0
    stacked = dataset.stack({'xy': ('x', 'y')})
    assert tuple(stacked.groupby('xy', squeeze=False).dims) == tuple(stacked.isel(xy=[0]).dims)
    assert len(recwarn) == 0