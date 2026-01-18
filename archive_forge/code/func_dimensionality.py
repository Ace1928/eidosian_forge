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
def dimensionality(obj):
    if isinstance(obj, (unit_registry.Quantity, unit_registry.Unit)):
        unit_like = obj
    else:
        unit_like = unit_registry.dimensionless
    return unit_like.dimensionality