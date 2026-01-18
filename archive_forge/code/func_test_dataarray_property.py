from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
@pytest.mark.parametrize('prop', ['attrs', 'chunks', 'coords', 'data', 'dims', 'dtype', 'encoding', 'imag', 'indexes', 'loc', 'name', 'nbytes', 'ndim', 'plot', 'real', 'shape', 'size', 'sizes', 'str', 'variable'])
def test_dataarray_property(prop):
    arr = make_xrarray({'x': 10, 'y': 5})
    getattr(arr, prop)