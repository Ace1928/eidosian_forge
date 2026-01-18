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
def array_attach_units(data, unit):
    if isinstance(data, Quantity):
        raise ValueError(f'cannot attach unit {unit} to quantity {data}')
    try:
        quantity = data * unit
    except np.core._exceptions.UFuncTypeError:
        if isinstance(unit, unit_registry.Unit):
            raise
        quantity = data
    return quantity