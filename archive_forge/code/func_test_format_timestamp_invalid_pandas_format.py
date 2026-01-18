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
def test_format_timestamp_invalid_pandas_format(self) -> None:
    expected = '2021-12-06 17:00:00 00'
    with pytest.raises(ValueError):
        formatting.format_timestamp(expected)