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
def test_inline_variable_array_repr_custom_repr() -> None:

    class CustomArray:

        def __init__(self, value, attr):
            self.value = value
            self.attr = attr

        def _repr_inline_(self, width):
            formatted = f'({self.attr}) {self.value}'
            if len(formatted) > width:
                formatted = f'({self.attr}) ...'
            return formatted

        def __array_namespace__(self, *args, **kwargs):
            return NotImplemented

        @property
        def shape(self) -> tuple[int, ...]:
            return self.value.shape

        @property
        def dtype(self):
            return self.value.dtype

        @property
        def ndim(self):
            return self.value.ndim
    value = CustomArray(np.array([20, 40]), 'm')
    variable = xr.Variable('x', value)
    max_width = 10
    actual = formatting.inline_variable_array_repr(variable, max_width=10)
    assert actual == value._repr_inline_(max_width)