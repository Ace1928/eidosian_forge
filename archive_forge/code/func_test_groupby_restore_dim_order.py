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
def test_groupby_restore_dim_order(self) -> None:
    array = DataArray(np.random.randn(5, 3), coords={'a': ('x', range(5)), 'b': ('y', range(3))}, dims=['x', 'y'])
    for by, expected_dims in [('x', ('x', 'y')), ('y', ('x', 'y')), ('a', ('a', 'y')), ('b', ('x', 'b'))]:
        result = array.groupby(by, squeeze=False).map(lambda x: x.squeeze())
        assert result.dims == expected_dims