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
def test_groupby_grouping_errors() -> None:
    dataset = xr.Dataset({'foo': ('x', [1, 1, 1])}, {'x': [1, 2, 3]})
    with pytest.raises(ValueError, match='None of the data falls within bins with edges'):
        dataset.groupby_bins('x', bins=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match='None of the data falls within bins with edges'):
        dataset.to_dataarray().groupby_bins('x', bins=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match='All bin edges are NaN.'):
        dataset.groupby_bins('x', bins=[np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match='All bin edges are NaN.'):
        dataset.to_dataarray().groupby_bins('x', bins=[np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match='Failed to group data.'):
        dataset.groupby(dataset.foo * np.nan)
    with pytest.raises(ValueError, match='Failed to group data.'):
        dataset.to_dataarray().groupby(dataset.foo * np.nan)