from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
def test_interpolate_invalid_interpolator_raises():
    da = xr.DataArray(np.array([1, 2, np.nan, 5], dtype=np.float64), dims='x')
    with pytest.raises(ValueError, match='not a valid'):
        da.interpolate_na(dim='x', method='foo')