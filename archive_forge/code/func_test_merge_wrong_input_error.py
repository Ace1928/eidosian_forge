from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_wrong_input_error(self):
    with pytest.raises(TypeError, match='objects must be an iterable'):
        xr.merge([1])
    ds = xr.Dataset(coords={'x': [1, 2]})
    with pytest.raises(TypeError, match='objects must be an iterable'):
        xr.merge({'a': ds})
    with pytest.raises(TypeError, match='objects must be an iterable'):
        xr.merge([ds, 1])