from __future__ import annotations
import functools
import operator
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, duck_array_ops
from xarray.tests import (
from xarray.tests.test_plot import PlotTestCase
from xarray.tests.test_variable import _PAD_XR_NP_ARGS
def strip_units(obj):
    if isinstance(obj, xr.Dataset):
        data_vars = {strip_units(name): strip_units(value) for name, value in obj.data_vars.items()}
        coords = {strip_units(name): strip_units(value) for name, value in obj.coords.items()}
        new_obj = xr.Dataset(data_vars=data_vars, coords=coords)
    elif isinstance(obj, xr.DataArray):
        data = array_strip_units(obj.variable._data)
        coords = {strip_units(name): (value.dims, array_strip_units(value.variable._data)) if isinstance(value.data, Quantity) else value for name, value in obj.coords.items()}
        new_obj = xr.DataArray(name=strip_units(obj.name), data=data, coords=coords, dims=obj.dims)
    elif isinstance(obj, xr.Variable):
        data = array_strip_units(obj.data)
        new_obj = obj.copy(data=data)
    elif isinstance(obj, unit_registry.Quantity):
        new_obj = obj.magnitude
    elif isinstance(obj, (list, tuple)):
        return type(obj)((strip_units(elem) for elem in obj))
    else:
        new_obj = obj
    return new_obj