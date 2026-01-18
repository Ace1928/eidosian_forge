from __future__ import annotations
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from xarray import (
from xarray.core import dtypes
from xarray.core.combine import (
from xarray.tests import assert_equal, assert_identical, requires_cftime
from xarray.tests.test_dataset import create_test_data
def test_combine_by_coords_mixed_unnamed_dataarrays(self):
    named_da = DataArray(name='a', data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x')
    unnamed_da = DataArray(data=[3.0, 4.0], coords={'x': [2, 3]}, dims='x')
    with pytest.raises(ValueError, match="Can't automatically combine unnamed DataArrays with"):
        combine_by_coords([named_da, unnamed_da])
    da = DataArray([0, 1], dims='x', coords={'x': [0, 1]})
    ds = Dataset({'x': [2, 3]})
    with pytest.raises(ValueError, match="Can't automatically combine unnamed DataArrays with"):
        combine_by_coords([da, ds])