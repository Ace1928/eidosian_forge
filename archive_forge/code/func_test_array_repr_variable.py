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
def test_array_repr_variable(self) -> None:
    var = xr.Variable('x', [0, 1])
    formatting.array_repr(var)
    with xr.set_options(display_expand_data=False):
        formatting.array_repr(var)