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
def test_groupby_multidim(self) -> None:
    array = self.make_groupby_multidim_example_array()
    for dim, expected_sum in [('lon', DataArray([5, 28, 23], coords=[('lon', [30.0, 40.0, 50.0])])), ('lat', DataArray([16, 40], coords=[('lat', [10.0, 20.0])]))]:
        actual_sum = array.groupby(dim).sum(...)
        assert_identical(expected_sum, actual_sum)