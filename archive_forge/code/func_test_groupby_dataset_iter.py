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
def test_groupby_dataset_iter() -> None:
    data = create_test_data()
    for n, (t, sub) in enumerate(list(data.groupby('dim1', squeeze=False))[:3]):
        assert data['dim1'][n] == t
        assert_equal(data['var1'][[n]], sub['var1'])
        assert_equal(data['var2'][[n]], sub['var2'])
        assert_equal(data['var3'][:, [n]], sub['var3'])