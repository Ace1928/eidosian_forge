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
def test_first_n_items(self) -> None:
    array = np.arange(100).reshape(10, 5, 2)
    for n in [3, 10, 13, 100, 200]:
        actual = formatting.first_n_items(array, n)
        expected = array.flat[:n]
        assert (expected == actual).all()
    with pytest.raises(ValueError, match='at least one item'):
        formatting.first_n_items(array, 0)