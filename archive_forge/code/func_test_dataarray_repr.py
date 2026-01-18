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
def test_dataarray_repr(self):
    a = xr.DataArray(sparse.COO.from_numpy(np.ones(4)), dims=['x'], coords={'y': ('x', sparse.COO.from_numpy(np.arange(4, dtype='i8')))})
    expected = dedent('            <xarray.DataArray (x: 4)> Size: 64B\n            <COO: shape=(4,), dtype=float64, nnz=4, fill_value=0.0>\n            Coordinates:\n                y        (x) int64 48B <COO: nnz=3, fill_value=0>\n            Dimensions without coordinates: x')
    assert expected == repr(a)