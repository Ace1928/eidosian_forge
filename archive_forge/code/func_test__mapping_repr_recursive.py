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
def test__mapping_repr_recursive() -> None:
    ds = xr.Dataset({'a': ('x', [1, 2, 3])})
    ds.attrs['ds'] = ds
    formatting.dataset_repr(ds)
    ds2 = xr.Dataset({'b': ('y', [1, 2, 3])})
    ds.attrs['ds'] = ds2
    ds2.attrs['ds'] = ds
    formatting.dataset_repr(ds2)