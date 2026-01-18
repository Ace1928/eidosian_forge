from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_dataarray_unnamed(self):
    data = xr.DataArray([1, 2], dims='x')
    with pytest.raises(ValueError, match='without providing an explicit name'):
        xr.merge([data])