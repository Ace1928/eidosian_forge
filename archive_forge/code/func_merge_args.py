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
def merge_args(default_args, new_args):
    from itertools import zip_longest
    fill_value = object()
    return [second if second is not fill_value else first for first, second in zip_longest(default_args, new_args, fillvalue=fill_value)]