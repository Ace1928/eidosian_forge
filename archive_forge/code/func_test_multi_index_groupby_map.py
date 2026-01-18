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
def test_multi_index_groupby_map(dataset) -> None:
    ds = dataset.isel(z=1, drop=True)[['foo']]
    expected = 2 * ds
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        actual = ds.stack(space=['x', 'y']).groupby('space').map(lambda x: 2 * x).unstack('space')
    assert_equal(expected, actual)