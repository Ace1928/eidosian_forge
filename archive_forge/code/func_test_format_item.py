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
def test_format_item(self) -> None:
    cases = [(pd.Timestamp('2000-01-01T12'), '2000-01-01T12:00:00'), (pd.Timestamp('2000-01-01'), '2000-01-01'), (pd.Timestamp('NaT'), 'NaT'), (pd.Timedelta('10 days 1 hour'), '10 days 01:00:00'), (pd.Timedelta('-3 days'), '-3 days +00:00:00'), (pd.Timedelta('3 hours'), '0 days 03:00:00'), (pd.Timedelta('NaT'), 'NaT'), ('foo', "'foo'"), (b'foo', "b'foo'"), (1, '1'), (1.0, '1.0'), (np.float16(1.1234), '1.123'), (np.float32(1.0111111), '1.011'), (np.float64(22.222222), '22.22')]
    for item, expected in cases:
        actual = formatting.format_item(item)
        assert expected == actual