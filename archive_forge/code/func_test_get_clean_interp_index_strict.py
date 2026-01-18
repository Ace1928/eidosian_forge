from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@pytest.mark.parametrize('index', ([0, 2, 1], [0, 1, 1]))
def test_get_clean_interp_index_strict(index):
    da = xr.DataArray([0, 1, 2], dims=('x',), coords={'x': index})
    with pytest.raises(ValueError):
        get_clean_interp_index(da, 'x')
    clean = get_clean_interp_index(da, 'x', strict=False)
    np.testing.assert_array_equal(index, clean)
    assert clean.dtype == np.float64