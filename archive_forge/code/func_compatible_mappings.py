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
def compatible_mappings(first, second):
    return {key: is_compatible(unit1, unit2) for key, (unit1, unit2) in zip_mappings(first, second)}