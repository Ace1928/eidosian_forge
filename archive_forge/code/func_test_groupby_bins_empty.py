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
def test_groupby_bins_empty(self) -> None:
    array = DataArray(np.arange(4), [('x', range(4))])
    bins = [0, 4, 5]
    bin_coords = pd.cut(array['x'], bins).categories
    actual = array.groupby_bins('x', bins).sum()
    expected = DataArray([6, np.nan], dims='x_bins', coords={'x_bins': bin_coords})
    assert_identical(expected, actual)
    assert len(array.x) == 4