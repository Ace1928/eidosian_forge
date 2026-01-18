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
@pytest.mark.parametrize('as_dataset', (False, True))
def test_format_xindexes_none(as_dataset: bool) -> None:
    expected = '    Indexes:\n        *empty*'
    expected = dedent(expected)
    obj: xr.DataArray | xr.Dataset = xr.DataArray()
    obj = obj._to_temp_dataset() if as_dataset else obj
    actual = repr(obj.xindexes)
    assert actual == expected