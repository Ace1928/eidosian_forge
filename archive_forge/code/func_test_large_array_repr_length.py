from __future__ import annotations
import sys
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray
import xarray as xr
from xarray.core import formatting
from xarray.tests import requires_cftime, requires_dask, requires_netCDF4
def test_large_array_repr_length() -> None:
    da = xr.DataArray(np.random.randn(100, 5, 1))
    result = repr(da).splitlines()
    assert len(result) < 50