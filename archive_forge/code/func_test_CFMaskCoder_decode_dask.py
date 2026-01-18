from __future__ import annotations
from contextlib import suppress
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding import variables
from xarray.conventions import decode_cf_variable, encode_cf_variable
from xarray.tests import assert_allclose, assert_equal, assert_identical, requires_dask
@requires_dask
def test_CFMaskCoder_decode_dask() -> None:
    original = xr.Variable(('x',), [0, -1, 1], {'_FillValue': -1}).chunk()
    expected = xr.Variable(('x',), [0, np.nan, 1])
    coder = variables.CFMaskCoder()
    encoded = coder.decode(original)
    assert isinstance(encoded.data, da.Array)
    assert_identical(expected, encoded)