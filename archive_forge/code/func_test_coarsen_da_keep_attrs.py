from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.core import duck_array_ops
from xarray.tests import (
@pytest.mark.parametrize('funcname, argument', [('reduce', (np.mean,)), ('mean', ())])
def test_coarsen_da_keep_attrs(funcname, argument) -> None:
    attrs_da = {'da_attr': 'test'}
    attrs_coords = {'attrs_coords': 'test'}
    data = np.linspace(10, 15, 100)
    coords = np.linspace(1, 10, 100)
    da = DataArray(data, dims='coord', coords={'coord': ('coord', coords, attrs_coords)}, attrs=attrs_da, name='name')
    func = getattr(da.coarsen(dim={'coord': 5}), funcname)
    result = func(*argument)
    assert result.attrs == attrs_da
    da.coord.attrs == attrs_coords
    assert result.name == 'name'
    func = getattr(da.coarsen(dim={'coord': 5}), funcname)
    result = func(*argument, keep_attrs=False)
    assert result.attrs == {}
    da.coord.attrs == {}
    assert result.name == 'name'
    func = getattr(da.coarsen(dim={'coord': 5}), funcname)
    with set_options(keep_attrs=False):
        result = func(*argument)
    assert result.attrs == {}
    da.coord.attrs == {}
    assert result.name == 'name'
    func = getattr(da.coarsen(dim={'coord': 5}), funcname)
    with set_options(keep_attrs=False):
        result = func(*argument, keep_attrs=True)
    assert result.attrs == attrs_da
    da.coord.attrs == {}
    assert result.name == 'name'
    func = getattr(da.coarsen(dim={'coord': 5}), funcname)
    with set_options(keep_attrs=True):
        result = func(*argument, keep_attrs=False)
    assert result.attrs == {}
    da.coord.attrs == {}
    assert result.name == 'name'