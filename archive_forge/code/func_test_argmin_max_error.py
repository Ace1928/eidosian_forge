from __future__ import annotations
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy import array, nan
from xarray import DataArray, Dataset, cftime_range, concat
from xarray.core import dtypes, duck_array_ops
from xarray.core.duck_array_ops import (
from xarray.namedarray.pycompat import array_type
from xarray.testing import assert_allclose, assert_equal, assert_identical
from xarray.tests import (
def test_argmin_max_error():
    da = construct_dataarray(2, np.bool_, contains_nan=True, dask=False)
    da[0] = np.nan
    with pytest.raises(ValueError):
        da.argmin(dim='y')