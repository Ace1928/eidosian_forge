from __future__ import annotations
from itertools import combinations, permutations
from typing import cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import _parse_array_of_cftime_strings
from xarray.core.types import InterpOptions
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
@requires_scipy
def test_dataset() -> None:
    ds = create_test_data()
    ds.attrs['foo'] = 'var'
    ds['var1'].attrs['buz'] = 'var2'
    new_dim2 = xr.DataArray([0.11, 0.21, 0.31], dims='z')
    interpolated = ds.interp(dim2=new_dim2)
    assert_allclose(interpolated['var1'], ds['var1'].interp(dim2=new_dim2))
    assert interpolated['var3'].equals(ds['var3'])
    interpolated['var1'][:, 1] = 1.0
    interpolated['var2'][:, 1] = 1.0
    interpolated['var3'][:, 1] = 1.0
    assert not interpolated['var1'].equals(ds['var1'])
    assert not interpolated['var2'].equals(ds['var2'])
    assert not interpolated['var3'].equals(ds['var3'])
    assert interpolated.attrs['foo'] == 'var'
    assert interpolated['var1'].attrs['buz'] == 'var2'