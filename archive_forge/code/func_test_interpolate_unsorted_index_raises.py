from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_scipy
def test_interpolate_unsorted_index_raises():
    vals = np.array([1, 2, 3], dtype=np.float64)
    expected = xr.DataArray(vals, dims='x', coords={'x': [2, 1, 3]})
    with pytest.raises(ValueError, match="Index 'x' must be monotonically increasing"):
        expected.interpolate_na(dim='x', method='index')